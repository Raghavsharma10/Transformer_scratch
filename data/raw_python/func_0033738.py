def format_hyperlink( val, hlx, hxl, xhl ):
    """
    Formats an html hyperlink into other forms.

    @hlx, hxl, xhl: values returned by set_output_format
    """
    if '<a href="' in str(val) and hlx != '<a href="':
        val = val.replace('<a href="', hlx).replace('">', hxl, 1).replace('</a>', xhl) 

    return val