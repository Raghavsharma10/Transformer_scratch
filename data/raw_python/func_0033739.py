def format_cell(val, round_floats = False, decimal_places = 2, format_links = False, 
    hlx = '', hxl = '', xhl = ''):
    """
    Applys smart_round and format_hyperlink to values in a cell if desired.
    """
    if round_floats:
        val = smart_round(val, decimal_places = decimal_places)
    if format_links:
        val = format_hyperlink(val, hlx, hxl, xhl)

    return val