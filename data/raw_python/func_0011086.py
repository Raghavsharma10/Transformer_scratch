def build_permutation_matrix(permutation):
    """Build a permutation matrix for a permutation.
    """
    matrix = lil_matrix((len(permutation), len(permutation)))
    column = 0
    for row in permutation:
        matrix[row, column] = 1
        column += 1
    return matrix