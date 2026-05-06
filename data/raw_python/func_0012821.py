def matchfieldnames(field_a, field_b):
    """Check match between two strings, ignoring case and spaces/underscores.
    
    Parameters
    ----------
    a : str
    b : str
    
    Returns
    -------
    bool
    
    """
    normalised_a = field_a.replace(' ', '_').lower()
    normalised_b = field_b.replace(' ', '_').lower()
    
    return normalised_a == normalised_b