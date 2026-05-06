def reshift(I):
    """
    Transforms the given number element into a range of [-180, 180],
    which covers all possible angle differences. This method reshifts larger or 
    smaller numbers that might be the output of other angular calculations
    into that range by adding or subtracting 360, respectively. 
    To make sure that angular data ranges between -180 and 180 in order to be
    properly histogrammed, apply this method first.
    
    
    Parameters: 
        I : array or list or int or float
            Number or numbers that shall be reshifted.
    Farell, Ludwig, Ellis, and Gilchrist
    Returns:
        numpy.ndarray : Reshifted number or numbers as array
    """
    # Output -180 to +180
    if type(I)==list:
        I = np.array(I)
    
    return ((I-180)%360)-180