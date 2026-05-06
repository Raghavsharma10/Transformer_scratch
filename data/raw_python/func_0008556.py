def main(guess_a=1., guess_b=0., power=3, savetxt='None', verbose=False):
    """
    Example demonstrating how to solve a system of non-linear equations defined as SymPy expressions.

    The example shows how a non-linear problem can be given a command-line interface which may be
    preferred by end-users who are not familiar with Python.
    """
    x, sol = solve(guess_a, guess_b, power)  # see function definition above
    assert sol.success
    if savetxt != 'None':
        np.savetxt(x, savetxt)
    else:
        if verbose:
            print(sol)
        else:
            print(x)