def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)