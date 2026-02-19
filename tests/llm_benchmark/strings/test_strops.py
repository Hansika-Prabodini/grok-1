"""Unit tests and benchmarks for string operations module."""

import pytest
import time
from src.llm_benchmark.strings.strops import StrOps


class TestCountChar:
    """Test cases for count_char method."""

    @pytest.mark.parametrize("s, c, expected", [
        # Normal cases
        ("hello world", "l", 3),
        ("hello world", "o", 2),
        ("hello world", "h", 1),
        
        # Not found cases
        ("hello world", "x", 0),
        ("hello world", "z", 0),
        
        # Empty inputs
        ("", "a", 0),
        ("hello", "", 0),
        ("", "", 0),
        
        # Single character
        ("a", "a", 1),
        ("b", "a", 0),
        
        # Multiple occurrences
        ("aaaa", "a", 4),
        ("abababab", "a", 4),
        ("abababab", "b", 4),
        
        # Case sensitivity
        ("Hello World", "h", 0),
        ("Hello World", "H", 1),
        ("Hello World", "l", 3),
        ("Hello World", "L", 0),
    ])
    def test_count_char(self, s, c, expected):
        """Test count_char with various inputs."""
        assert StrOps.count_char(s, c) == expected


class TestSubstringSearch:
    """Test cases for substring_search method."""

    @pytest.mark.parametrize("s, sub, expected", [
        # Normal cases
        ("hello world", "world", [6]),
        ("hello world", "hello", [0]),
        ("hello world", "o", [4, 7]),
        ("hello world", "l", [2, 3, 9]),
        
        # Not found cases
        ("hello world", "xyz", []),
        ("hello world", "python", []),
        
        # Empty inputs
        ("", "test", []),
        ("hello", "", []),
        ("", "", []),
        
        # Single character
        ("a", "a", [0]),
        ("abc", "a", [0]),
        ("abc", "c", [2]),
        
        # Overlapping occurrences
        ("aaa", "aa", [0, 1]),
        ("aaaaa", "aaa", [0, 1, 2]),
        ("ababa", "aba", [0, 2]),
        ("abababa", "aba", [0, 2, 4]),
        
        # Multiple non-overlapping occurrences
        ("abc abc abc", "abc", [0, 4, 8]),
        ("hello hello", "hello", [0, 6]),
        
        # Substring equals string
        ("test", "test", [0]),
        
        # Substring longer than string
        ("hi", "hello", []),
        
        # Case sensitivity
        ("Hello World", "hello", []),
        ("Hello World", "Hello", [0]),
        ("Hello World", "World", [6]),
        ("Hello World", "world", []),
    ])
    def test_substring_search(self, s, sub, expected):
        """Test substring_search with various inputs."""
        assert StrOps.substring_search(s, sub) == expected


def test_benchmark_count_char():
    """Benchmark test for count_char method."""
    # Use a moderate length string for benchmarking
    test_string = "The quick brown fox jumps over the lazy dog. " * 100
    test_char = "o"
    
    start_time = time.perf_counter()
    result = StrOps.count_char(test_string, test_char)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # Verify the function works correctly
    assert result > 0
    assert isinstance(result, int)
    
    # Print timing information (can be captured by pytest)
    print(f"\ncount_char benchmark: {execution_time:.6f} seconds")
    print(f"String length: {len(test_string)} characters")
    print(f"Character count: {result}")
    
    # Assert execution time is reasonable (should be very fast)
    assert execution_time < 1.0, "count_char took too long"


def test_benchmark_substring_search():
    """Benchmark test for substring_search method."""
    # Use a moderate length string for benchmarking
    test_string = "The quick brown fox jumps over the lazy dog. " * 100
    test_substring = "fox"
    
    start_time = time.perf_counter()
    result = StrOps.substring_search(test_string, test_substring)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # Verify the function works correctly
    assert len(result) > 0
    assert isinstance(result, list)
    assert all(isinstance(pos, int) for pos in result)
    
    # Print timing information (can be captured by pytest)
    print(f"\nsubstring_search benchmark: {execution_time:.6f} seconds")
    print(f"String length: {len(test_string)} characters")
    print(f"Substring occurrences found: {len(result)}")
    print(f"Positions: {result[:10]}...")  # Show first 10 positions
    
    # Assert execution time is reasonable
    assert execution_time < 1.0, "substring_search took too long"
