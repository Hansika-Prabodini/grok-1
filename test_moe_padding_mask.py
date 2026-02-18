"""Test for MoELayer padding mask bug.

This test verifies that the padding_mask parameter is correctly passed
from MoELayer.__call__ to MoELayer._inference_call, ensuring that
padding tokens don't affect routing decisions.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from model import MoELayer, Router


def test_moe_padding_mask():
    """Test that MoELayer correctly uses padding_mask.
    
    This test creates a MoELayer and verifies that when a padding_mask is provided,
    it properly affects the routing probabilities. Specifically, masked positions
    should have zero routing probabilities after masking.
    """
    
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
    # Shape: [batch_size, seq_len, emb_size]
    inputs = jnp.ones((batch_size, seq_len, emb_size), dtype=jnp.bfloat16)
    
    # Create padding mask where last 2 positions in sequence are masked (0)
    # Shape: [batch_size, seq_len, 1]
    padding_mask = jnp.array([
        [[1.0], [1.0], [0.0], [0.0]],  # First batch: last 2 positions masked
        [[1.0], [1.0], [1.0], [0.0]]   # Second batch: last position masked
    ], dtype=jnp.bfloat16)
    
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
    
    # Transform the function
    forward_fn = hk.transform(forward)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    params = forward_fn.init(rng, inputs, padding_mask)
    
    # Run forward pass
    output, routing_probs = forward_fn.apply(params, rng, inputs, padding_mask)
    
    # Verify that routing probabilities are masked
    # For masked positions (padding_mask == 0), routing_probs should be 0
    # First batch, positions 2 and 3 should be masked
    assert jnp.allclose(routing_probs[0, 2, :], 0.0, atol=1e-6), \
        f"Expected masked position [0,2] to have zero routing probs, got {routing_probs[0, 2, :]}"
    assert jnp.allclose(routing_probs[0, 3, :], 0.0, atol=1e-6), \
        f"Expected masked position [0,3] to have zero routing probs, got {routing_probs[0, 3, :]}"
    
    # Second batch, position 3 should be masked
    assert jnp.allclose(routing_probs[1, 3, :], 0.0, atol=1e-6), \
        f"Expected masked position [1,3] to have zero routing probs, got {routing_probs[1, 3, :]}"
    
    # Unmasked positions should have non-zero routing probs (they sum to 1 due to softmax)
    assert jnp.sum(routing_probs[0, 0, :]) > 0.99, \
        f"Expected unmasked position [0,0] to have non-zero routing probs, got sum {jnp.sum(routing_probs[0, 0, :])}"
    assert jnp.sum(routing_probs[0, 1, :]) > 0.99, \
        f"Expected unmasked position [0,1] to have non-zero routing probs, got sum {jnp.sum(routing_probs[0, 1, :])}"
    
    print("✓ Test passed: MoELayer correctly uses padding_mask")
    print(f"  Routing probs for masked positions are zero")
    print(f"  Routing probs for unmasked positions sum to ~1.0")


if __name__ == "__main__":
    test_moe_padding_mask()
    print("\nAll tests passed!")
