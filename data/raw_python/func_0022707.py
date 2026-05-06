def compact(vertices, indices, tolerance=1e-3):
    """ Compact vertices and indices within given tolerance """

    # Transform vertices into a structured array for np.unique to work
    n = len(vertices)
    V = np.zeros(n, dtype=[("pos", np.float32, 3)])
    V["pos"][:, 0] = vertices[:, 0]
    V["pos"][:, 1] = vertices[:, 1]
    V["pos"][:, 2] = vertices[:, 2]

    epsilon = 1e-3
    decimals = int(np.log(epsilon)/np.log(1/10.))

    # Round all vertices within given decimals
    V_ = np.zeros_like(V)
    X = V["pos"][:, 0].round(decimals=decimals)
    X[np.where(abs(X) < epsilon)] = 0

    V_["pos"][:, 0] = X
    Y = V["pos"][:, 1].round(decimals=decimals)
    Y[np.where(abs(Y) < epsilon)] = 0
    V_["pos"][:, 1] = Y

    Z = V["pos"][:, 2].round(decimals=decimals)
    Z[np.where(abs(Z) < epsilon)] = 0
    V_["pos"][:, 2] = Z

    # Find the unique vertices AND the mapping
    U, RI = np.unique(V_, return_inverse=True)

    # Translate indices from original vertices into the reduced set (U)
    indices = indices.ravel()
    I_ = indices.copy().ravel()
    for i in range(len(indices)):
        I_[i] = RI[indices[i]]
    I_ = I_.reshape(len(indices)/3, 3)

    # Return reduced vertices set, transalted indices and mapping that allows
    # to go from U to V
    return U.view(np.float32).reshape(len(U), 3), I_, RI