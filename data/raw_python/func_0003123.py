def html_elem(e, ct, withtype=False):
    """
    Format a result element as an HTML table cell.
      @param e (list): a pair \c (value,type)
      @param ct (str): cell type (th or td)
      @param withtype (bool): add an additional cell with the element type
    """
    # Header cell
    if ct == 'th':
        return '<th>{0}</th><th>{1}</th>'.format(*e) if withtype else '<th>{}</th>'.format(e)
    # Content cell
    if e[1] in ('uri', 'URIRef'):
        html = u'<{0} class=val><a href="{1}" target="_other">{2}</a></{0}>'.format(ct, e[0], escape(e[0]))
    else:
        html = u'<{0} class=val>{1}</{0}>'.format(ct, escape(e[0]))
    # Create the optional cell for the type
    if withtype:
        html += u'<{0} class=typ>{1}</{0}>'.format(ct, e[1])
    return html