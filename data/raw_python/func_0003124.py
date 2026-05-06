def html_table(data, header=True, limit=None, withtype=False):
    """
    Return a double iterable as an HTML table
      @param data (iterable): the data to format
      @param header (bool): if the first row is a header row
      @param limit (int): maximum number of rows to render (excluding header)
      @param withtype (bool): if columns are to have an alternating CSS class
        (even/odd) or not.
      @return (int,string): a pair <number-of-rendered-rows>, <html-table>
    """
    if header and limit:
        limit += 1
    ct = 'th' if header else 'td'
    rc = 'hdr' if header else 'odd'

    # import codecs
    # import datetime
    # with codecs.open( '/tmp/dump', 'w', encoding='utf-8') as f:
    #     print( '************', datetime.datetime.now(), file=f )
    #     for n, row in enumerate(data):
    #         print( '-------', n,  file=f )
    #         for n, c in enumerate(row):
    #             print( type(c), repr(c), file=f )

    html = u'<table>'
    rn = -1
    for rn, row in enumerate(data):
        html += u'<tr class={}>'.format(rc)
        html += '\n'.join((html_elem(c, ct, withtype) for c in row))
        html += u'</tr>'
        rc = 'even' if rc == 'odd' else 'odd'
        ct = 'td'
        if limit:
            limit -= 1
            if not limit:
                break
    return (0, '') if rn < 0 else (rn+1-header, html+u'</table>')