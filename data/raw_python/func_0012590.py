def snip_string(string, max_len=20, snip_string='...', snip_point=0.5):
    """
    Snips a string so that it is no longer than max_len, replacing deleted
    characters with the snip_string.
    The snip is done at snip_point, which is a fraction between 0 and 1,
    indicating relatively where along the string to snip. snip_point of
    0.5 would be the middle.
    >>> snip_string('this is long', 8)
    'this ...'
    >>> snip_string('this is long', 8, snip_point=0.5)
    'th...ong'
    >>> snip_string('this is long', 12)
    'this is long'
    >>> snip_string('this is long', 8, '~')
    'this is~'
    >>> snip_string('this is long', 8, '~', 0.5)
    'thi~long'
    
    """
    if len(string) <= max_len:
        new_string = string
    else:
        visible_len = (max_len - len(snip_string))
        start_len = int(visible_len*snip_point)
        end_len = visible_len-start_len
        
        new_string = string[0:start_len]+ snip_string
        if end_len > 0:
            new_string += string[-end_len:]
    
    return new_string