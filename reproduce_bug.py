"""
Unit test to verify the fix for the MultiHeadAttention syntax error bug.

This test imports and creates a MultiHeadAttention instance, which would fail
before the patch due to the syntax error on line 704 of model.py where
`attn_output_multiplier: 1.0,` was missing the type annotation.

The bug: attn_output_multiplier: 1.0,
The fix: attn_output_multiplier: float = 1.0,
"""

import sys


def test_multiheadattention_initialization():
    """Test that MultiHeadAttention can be properly initialized.
    
    This test will fail before the patch because of the syntax error in the
    parameter definition, and will pass after the patch.
    """
    try:
        # This import will fail if there's a syntax error in model.py
        from model import MultiHeadAttention
        
        # Try to create an instance with default parameters
        # Note: We can't actually call it without haiku context, but we can test initialization
        mha = MultiHeadAttention(
            num_q_heads=8,
            num_kv_heads=4,
            key_size=64,
        )
        
        # Verify the default value is set correctly
        assert hasattr(mha, 'attn_output_multiplier'), "attn_output_multiplier attribute should exist"
        assert mha.attn_output_multiplier == 1.0, "Default attn_output_multiplier should be 1.0"
        
        # Test with custom value
        mha_custom = MultiHeadAttention(
            num_q_heads=8,
            num_kv_heads=4,
            key_size=64,
            attn_output_multiplier=0.5,
        )
        assert mha_custom.attn_output_multiplier == 0.5, "Custom attn_output_multiplier should be 0.5"
        
        print("✓ Test passed: MultiHeadAttention initialization works correctly")
        return True
        
    except SyntaxError as e:
        print(f"✗ Test failed: Syntax error in model.py - {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = test_multiheadattention_initialization()
    sys.exit(0 if success else 1)
