def bigint_to_string(val):
    """ Converts @val to a string if it is a big integer (|>2**53-1|)

        @val: #int or #float

        -> #str if @val is a big integer, otherwise @val
    """
    if isinstance(val, _NUMBERS) and not abs(val) <= 2**53-1:
        return str(val)
    return val