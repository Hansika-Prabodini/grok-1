"""Test for MultiHeadAttention initialization bug.

This test verifies that the MultiHeadAttention class can be properly instantiated
with the attn_output_multiplier parameter. 

BEFORE THE FIX: This test would fail with a TypeError because line 704 in model.py
has invalid syntax: `attn_output_multiplier: 1.0,` which tries to use a literal
value (1.0) as a type annotation.

AFTER THE FIX: This test passes because the syntax is corrected to:
`attn_output_multiplier: float = 1.0,` which properly annotates the type and
provides a default value.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from model import MultiHeadAttention


def test_multihead_attention_default_multiplier():
    """Test that MultiHeadAttention can be instantiated with default attn_output_multiplier.
    
    This tests the fix for the syntax error on line 704 of model.py where
    `attn_output_multiplier: 1.0,` should be `attn_output_multiplier: float = 1.0,`
    """
    
    # Test parameters
    num_q_heads = 4
    num_kv_heads = 2
    key_size = 8
    batch_size = 2
    seq_len = 4
    model_size = num_q_heads * key_size
    
    # Create test inputs
    query = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    key = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    value = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    
    def forward(query, key, value):
        # This should work - instantiate with default attn_output_multiplier
        mha = MultiHeadAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_size=key_size,
            # Not passing attn_output_multiplier, should use default value of 1.0
        )
        
        # Verify the default value is set correctly
        assert hasattr(mha, 'attn_output_multiplier'), "Missing attn_output_multiplier attribute"
        assert mha.attn_output_multiplier == 1.0, f"Expected default value 1.0, got {mha.attn_output_multiplier}"
        
        output = mha(query, key, value)
        return output
    
    # Transform and initialize
    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(42)
    
    try:
        params = forward_fn.init(rng, query, key, value)
        output = forward_fn.apply(params, rng, query, key, value)
        print("✓ PASS: MultiHeadAttention instantiated successfully with default attn_output_multiplier=1.0")
        return True
    except TypeError as e:
        print(f"❌ FAIL: TypeError during instantiation: {e}")
        print("This indicates the syntax error on line 704 is still present.")
        raise


def test_multihead_attention_custom_multiplier():
    """Test that MultiHeadAttention can be instantiated with custom attn_output_multiplier."""
    
    # Test parameters
    num_q_heads = 4
    num_kv_heads = 2
    key_size = 8
    batch_size = 2
    seq_len = 4
    model_size = num_q_heads * key_size
    custom_multiplier = 0.5
    
    # Create test inputs
    query = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    key = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    value = jnp.ones((batch_size, seq_len, model_size), dtype=jnp.bfloat16)
    
    def forward(query, key, value):
        # Instantiate with custom attn_output_multiplier
        mha = MultiHeadAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_size=key_size,
            attn_output_multiplier=custom_multiplier,
        )
        
        # Verify the custom value is set correctly
        assert mha.attn_output_multiplier == custom_multiplier, \
            f"Expected custom value {custom_multiplier}, got {mha.attn_output_multiplier}"
        
        output = mha(query, key, value)
        return output
    
    # Transform and initialize
    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(42)
    
    try:
        params = forward_fn.init(rng, query, key, value)
        output = forward_fn.apply(params, rng, query, key, value)
        print(f"✓ PASS: MultiHeadAttention instantiated successfully with custom attn_output_multiplier={custom_multiplier}")
        return True
    except TypeError as e:
        print(f"❌ FAIL: TypeError during instantiation: {e}")
        print("This indicates the syntax error on line 704 is still present.")
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("Testing MultiHeadAttention Initialization Bug Fix")
    print("=" * 80)
    print("\nTest 1: Default attn_output_multiplier")
    print("-" * 80)
    
    success1 = test_multihead_attention_default_multiplier()
    
    print("\nTest 2: Custom attn_output_multiplier")
    print("-" * 80)
    
    success2 = test_multihead_attention_custom_multiplier()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("\nThe attn_output_multiplier parameter is now correctly defined with")
        print("proper syntax: attn_output_multiplier: float = 1.0")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
