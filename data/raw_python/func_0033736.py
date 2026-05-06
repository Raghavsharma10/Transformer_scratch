def set_output_format( output_format ):
    """
    Sets output format; returns standard bits of table. These are:
        ttx: how to start a title for a set of tables
        xtt: how to end a title for a set of tables
        tx: how to start a table
        xt: how to close a table
        capx: how to start a caption for the table
        xcap: how to close a caption for the table
        rx: how to start a row and the first cell in the row
        xr: how to close a row and the last cell in the row
        rspx: how to start a cell with a row span argument
        xrsp: how to close the row span argument
        cx: how to open a cell
        xc: how to close a cell
    """
    if output_format == 'wiki':
        ttx = '== '
        xtt = ' =='
        tx = ''
        xt = ''
        capx = "'''"
        xcap = "'''"
        rx = '|'
        xr = '|'
        rspx = '|<|'
        xrsp = '>'
        cx = '|'
        xc = '|'
        hlx = '['
        hxl = ' '
        xhl = ']'

    elif output_format == "html":
        ttx = '<b>'
        xtt = '</b><hr>'
        tx = '<table border = "1">'
        xt = '</table><br><br>'
        capx = '<caption>'
        xcap = '</caption>'
        rx = '<tr>'
        xr = '</tr>'
        rspx = '<td rowspan='
        xrsp = '>'
        cx = '<td>'
        xc = '</td>'
        hlx = '<a href="'
        hxl = '">'
        xhl = "</a>"

    else:
        raise ValueError("unrecognized output_format %s" % output_format)

    return ttx, xtt, tx, xt, capx, xcap, rx, xr, cx, xc, rspx, xrsp, hlx, hxl, xhl