def solve(guess_a, guess_b, power, solver='scipy'):
    """ Constructs a pyneqsys.symbolic.SymbolicSys instance and returns from its ``solve`` method. """
    # The problem is 2 dimensional so we need 2 symbols
    x = sp.symbols('x:2', real=True)
    # There is a user specified parameter ``p`` in this problem:
    p = sp.Symbol('p', real=True, negative=False, integer=True)
    # Our system consists of 2-non-linear equations:
    f = [x[0] + (x[0] - x[1])**p/2 - 1,
         (x[1] - x[0])**p/2 + x[1]]
    # We construct our ``SymbolicSys`` instance by passing variables, equations and parameters:
    neqsys = SymbolicSys(x, f, [p])  # (this will derive the Jacobian symbolically)

    # Finally we solve the system using user-specified ``solver`` choice:
    return neqsys.solve([guess_a, guess_b], [power], solver=solver)