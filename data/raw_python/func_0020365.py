def fit(data, guess_angles=None):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points 

    best_fit = None
    best_score = float('inf')

    for sp in start_points:
        fitted = minimize(lambda x : G(direction(x[0], x[1]), Xs),
                    sp, method='Powell', tol=1e-6)

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun