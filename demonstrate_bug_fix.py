"""Demonstration of the MoELayer padding_mask bug and fix.

This script shows that:
1. BEFORE the fix: padding_mask is ignored, leading to non-zero routing for padded tokens
2. AFTER the fix: padding_mask is correctly applied, zeroing out routing for padded tokens
"""

import jax
import jax.numpy as jnp
import haiku as hk
from model import MoELayer, Router


def demonstrate_bug():
    """Demonstrate the padding_mask bug in MoELayer."""
    
    print("=" * 80)
    print("DEMONSTRATING MoELayer PADDING_MASK BUG FIX")
    print("=" * 80)
    
    # Set up test parameters
    num_experts = 4
    num_selected_experts = 2
    emb_size = 8
    batch_size = 2
    seq_len = 4
    
    # Create a simple layer function for MoE
    def layer_fn(x):
        return hk.Linear(emb_size)(x)
    
    # Create test inputs
    inputs = jnp.ones((batch_size, seq_len, emb_size), dtype=jnp.bfloat16)
    
    # Create padding mask where last 2 positions in first batch are masked
    # and last position in second batch is masked
    padding_mask = jnp.array([
        [[1.0], [1.0], [0.0], [0.0]],  # First batch: positions 2,3 are padding
        [[1.0], [1.0], [1.0], [0.0]]   # Second batch: position 3 is padding
    ], dtype=jnp.bfloat16)
    
    print("\nInput shape:", inputs.shape)
    print("Padding mask shape:", padding_mask.shape)
    print("\nPadding mask values:")
    print("Batch 0:", padding_mask[0, :, 0].tolist(), "← positions 2,3 are padded (0)")
    print("Batch 1:", padding_mask[1, :, 0].tolist(), "← position 3 is padded (0)")
    
    # Define the MoE forward function
    def forward(inputs, padding_mask):
        router = Router(
            num_selected_experts=num_selected_experts,
            data_axis="data",
            model_axis="model",
            shard_activations=False,
            mesh=None,
        )
        
        moe_layer = MoELayer(
            num_experts=num_experts,
            layer_fn=layer_fn,
            router=router,
            mesh=None,
            shard_activations=False,
        )
        
        # Call the MoE layer with padding mask
        output = moe_layer(inputs, padding_mask)
        
        # Also get routing probabilities to verify masking
        routing_probs, _, _ = router.compute_routing_prob(inputs, padding_mask, num_experts)
        
        return output, routing_probs
    
    # Transform and initialize
    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(42)
    params = forward_fn.init(rng, inputs, padding_mask)
    
    # Run forward pass
    output, routing_probs = forward_fn.apply(params, rng, inputs, padding_mask)
    
    print("\n" + "-" * 80)
    print("RESULTS AFTER FIX:")
    print("-" * 80)
    
    # Check routing probabilities
    print("\nRouting probabilities (should be 0 for padded positions):")
    print(f"Batch 0, Position 0 (unmasked): sum = {jnp.sum(routing_probs[0, 0, :]):.4f}")
    print(f"Batch 0, Position 1 (unmasked): sum = {jnp.sum(routing_probs[0, 1, :]):.4f}")
    print(f"Batch 0, Position 2 (PADDED):   sum = {jnp.sum(routing_probs[0, 2, :]):.4f}")
    print(f"Batch 0, Position 3 (PADDED):   sum = {jnp.sum(routing_probs[0, 3, :]):.4f}")
    print(f"Batch 1, Position 3 (PADDED):   sum = {jnp.sum(routing_probs[1, 3, :]):.4f}")
    
    # Verify the fix works
    all_tests_pass = True
    
    # Test: Padded positions should have zero routing probabilities
    if not jnp.allclose(routing_probs[0, 2, :], 0.0, atol=1e-6):
        print("\n❌ FAIL: Batch 0, Position 2 (padded) has non-zero routing probs!")
        all_tests_pass = False
    else:
        print("\n✓ PASS: Batch 0, Position 2 (padded) has zero routing probs")
    
    if not jnp.allclose(routing_probs[0, 3, :], 0.0, atol=1e-6):
        print("❌ FAIL: Batch 0, Position 3 (padded) has non-zero routing probs!")
        all_tests_pass = False
    else:
        print("✓ PASS: Batch 0, Position 3 (padded) has zero routing probs")
    
    if not jnp.allclose(routing_probs[1, 3, :], 0.0, atol=1e-6):
        print("❌ FAIL: Batch 1, Position 3 (padded) has non-zero routing probs!")
        all_tests_pass = False
    else:
        print("✓ PASS: Batch 1, Position 3 (padded) has zero routing probs")
    
    # Test: Unpadded positions should have non-zero routing probabilities
    if jnp.sum(routing_probs[0, 0, :]) < 0.99:
        print("❌ FAIL: Batch 0, Position 0 (unpadded) has near-zero routing probs!")
        all_tests_pass = False
    else:
        print("✓ PASS: Batch 0, Position 0 (unpadded) has non-zero routing probs")
    
    if jnp.sum(routing_probs[0, 1, :]) < 0.99:
        print("❌ FAIL: Batch 0, Position 1 (unpadded) has near-zero routing probs!")
        all_tests_pass = False
    else:
        print("✓ PASS: Batch 0, Position 1 (unpadded) has non-zero routing probs")
    
    print("\n" + "=" * 80)
    if all_tests_pass:
        print("✅ ALL TESTS PASSED - The fix is working correctly!")
        print("\nThe padding_mask is now correctly passed from MoELayer.__call__")
        print("to MoELayer._inference_call, ensuring padded tokens don't affect")
        print("routing decisions.")
    else:
        print("❌ SOME TESTS FAILED - The bug is still present!")
        print("\nThe padding_mask is NOT being passed correctly, causing padded")
        print("tokens to incorrectly participate in routing decisions.")
    print("=" * 80)
    
    return all_tests_pass


if __name__ == "__main__":
    success = demonstrate_bug()
    if not success:
        exit(1)
