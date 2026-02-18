"""Unit test for MultiHeadAttention type annotation bug fix.

This test verifies that the MultiHeadAttention class can be properly
instantiated with the attn_output_multiplier parameter. Before the fix,
line 704 in model.py had a syntax error:
    attn_output_multiplier: 1.0,
which should have been:
    attn_output_multiplier: float = 1.0,

The bug would prevent the file from being imported at all due to invalid syntax.
"""

import unittest
import jax
import jax.numpy as jnp
import haiku as hk


class TestMultiHeadAttentionBugFix(unittest.TestCase):
    """Test that the MultiHeadAttention class can be properly instantiated."""

    def test_import_model_module(self):
        """Test that the model module can be imported without syntax errors."""
        try:
            import model
            # If we can import the module, the syntax error is fixed
            self.assertTrue(True)
        except SyntaxError as e:
            self.fail(f"model.py has a syntax error: {e}")

    def test_multihead_attention_instantiation(self):
        """Test that MultiHeadAttention can be instantiated with default parameters."""
        from model import MultiHeadAttention
        
        # This should work without errors after the fix
        try:
            mha = MultiHeadAttention(
                num_q_heads=8,
                num_kv_heads=8,
                key_size=64,
            )
            self.assertIsNotNone(mha)
            # Verify the default value is set correctly
            self.assertEqual(mha.attn_output_multiplier, 1.0)
        except TypeError as e:
            self.fail(f"MultiHeadAttention instantiation failed: {e}")

    def test_multihead_attention_custom_multiplier(self):
        """Test that MultiHeadAttention can be instantiated with a custom attn_output_multiplier."""
        from model import MultiHeadAttention
        
        custom_multiplier = 0.5
        try:
            mha = MultiHeadAttention(
                num_q_heads=8,
                num_kv_heads=8,
                key_size=64,
                attn_output_multiplier=custom_multiplier,
            )
            self.assertIsNotNone(mha)
            # Verify the custom value is set correctly
            self.assertEqual(mha.attn_output_multiplier, custom_multiplier)
        except TypeError as e:
            self.fail(f"MultiHeadAttention instantiation with custom multiplier failed: {e}")

    def test_multihead_attention_in_transform(self):
        """Test that MultiHeadAttention works within a haiku transform."""
        from model import MultiHeadAttention
        
        def forward_fn(x):
            mha = MultiHeadAttention(
                num_q_heads=4,
                num_kv_heads=4,
                key_size=32,
                attn_output_multiplier=0.8,
            )
            # Create dummy inputs
            batch_size, seq_len, emb_dim = x.shape
            query = x
            key = x
            value = x
            mask = jnp.ones((batch_size, 1, seq_len, seq_len))
            
            output = mha(query, key, value, mask=mask)
            return output.embeddings
        
        # Transform the function
        transformed = hk.transform(forward_fn)
        
        # Initialize with dummy data
        dummy_input = jnp.ones((2, 10, 128))
        rng = jax.random.PRNGKey(42)
        
        try:
            params = transformed.init(rng, dummy_input)
            self.assertIsNotNone(params)
            
            # Apply the function
            output = transformed.apply(params, rng, dummy_input)
            self.assertIsNotNone(output)
            self.assertEqual(output.shape[0], 2)  # batch size
            self.assertEqual(output.shape[1], 10)  # sequence length
        except Exception as e:
            self.fail(f"MultiHeadAttention failed in haiku transform: {e}")


if __name__ == '__main__':
    unittest.main()
