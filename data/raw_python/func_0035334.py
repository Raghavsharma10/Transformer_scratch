def _to_tonnetz(chromagram):
    """Project a chromagram on the tonnetz.

    Returned value is normalized to prevent numerical instabilities.  
    """
    if np.sum(np.abs(chromagram)) == 0.:
        # The input is an empty chord, return zero. 
        return np.zeros(6)
    
    _tonnetz = np.dot(__TONNETZ_MATRIX, chromagram)
    one_norm = np.sum(np.abs(_tonnetz))  # Non-zero value
    _tonnetz = _tonnetz / float(one_norm) # Normalize tonnetz vector
    return _tonnetz