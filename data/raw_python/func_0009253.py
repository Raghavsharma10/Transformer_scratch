def _can_use_fast_algorithm(x, y, exponent=1):
    """
    Check if the fast algorithm for distance stats can be used.

    The fast algorithm has complexity :math:`O(NlogN)`, better than the
    complexity of the naive algorithm (:math:`O(N^2)`).

    The algorithm can only be used for random variables (not vectors) where
    the number of instances is greater than 3. Also, the exponent must be 1.

    """
    return (_is_random_variable(x) and _is_random_variable(y) and
            x.shape[0] > 3 and y.shape[0] > 3 and exponent == 1)