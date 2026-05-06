def calc_steady_state_dist(R):
    """Calculate the steady state dist of a 4 state markov transition matrix.

    Parameters
    ----------
    R : ndarray
        Markov transition matrix

    Returns
    -------
    p_ss : ndarray
        Steady state probability distribution
        
    """
    #Calc steady state distribution for a dinucleotide bias matrix
    
    w, v = np.linalg.eig(R)
    
    for i in range(4):
        if np.abs(w[i] - 1) < 1e-8:
            return np.real(v[:, i] / np.sum(v[:, i]))
    return -1