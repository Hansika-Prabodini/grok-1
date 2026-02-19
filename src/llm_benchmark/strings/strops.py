"""String operations module for LLM benchmark.

This module provides fundamental string manipulation functions for benchmarking.
"""

from typing import List


class StrOps:
    """Collection of static methods for string operations."""

    @staticmethod
    def count_char(s: str, c: str) -> int:
        """Count the number of times a character appears in a string.

        Args:
            s: The input string to search in.
            c: The character to count (case-sensitive).

        Returns:
            The number of times character c appears in string s.
            Returns 0 if character not found or if inputs are empty.
        """
        if not s or not c:
            return 0
        return s.count(c)

    @staticmethod
    def substring_search(s: str, sub: str) -> List[int]:
        """Find all starting positions where a substring appears in a string.

        Args:
            s: The input string to search in.
            sub: The substring to search for.

        Returns:
            A list of integer indices (0-based) where the substring starts.
            Returns empty list if substring not found or if inputs are empty.
            Handles overlapping occurrences (e.g., "aaa" in "aaaaa" returns [0, 1, 2]).
        """
        if not s or not sub:
            return []
        
        positions = []
        start = 0
        
        while True:
            pos = s.find(sub, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1  # Move by 1 to find overlapping matches
        
        return positions
