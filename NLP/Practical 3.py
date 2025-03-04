import numpy as np


def edit_distance_np(s1, s2):
    """
    Compute the minimum edit (Levenshtein) distance between two strings s1 and s2 using NumPy.
    This distance is the minimum number of operations (insertions, deletions, substitutions)
    required to transform s1 into s2.
    """
    m, n = len(s1), len(s2)
    # Initialize the DP matrix using NumPy
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Fill the first row and column.z
    dp[0, :] = np.arange(n + 1)
    dp[:, 0] = np.arange(m + 1)

    # Compute the edit distances.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j],  # deletion
                    dp[i, j - 1],  # insertion
                    dp[i - 1, j - 1]  # substitution
                )
    return dp[m, n]


def correct_spelling_np(word, dictionary):
    """
    Given an input word and a dictionary (list of words), find the dictionary word
    with the smallest edit distance to the input word.
    """
    # Calculate edit distance for each candidate in the dictionary
    distances = [(dict_word, edit_distance_np(word, dict_word)) for dict_word in dictionary]
    # Sort candidates based on edit distance and return the best match
    distances.sort(key=lambda x: x[1])
    return distances[0]


# Example usage:
dictionary = ["kitten", "sitting", "bitten", "written", "kitchen"]
input_word = "kittn"  # A misspelling of "kitten"

# Find the best correction using the NumPy-based edit distance
correction, distance = correct_spelling_np(input_word, dictionary)
print(f"Did you mean '{correction}'? (Edit distance: {distance})")

# Testing the edit_distance_np function directly:
print("Edit distance between 'kitten' and 'sitting':", edit_distance_np("kitten", "sitting"))
