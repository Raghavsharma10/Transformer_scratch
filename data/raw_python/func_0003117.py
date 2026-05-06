def data_msglist( msglist ):
    """
    Return a Jupyter display_data message, in both HTML & text formats, by 
    joining together all passed messages.

      @param msglist (iterable): an iterable containing a list of tuples
        (message, css_style)      

    Each message is either a text string, or a list. In the latter case it is
    assumed to be a format string + parameters.
    """
    txt = html = u''
    for msg, css in msglist:
        if is_collection(msg):
            msg = msg[0].format(*msg[1:])
        html += div( escape(msg).replace('\n','<br/>'), css=css or 'msg' )
        txt += msg + "\n"
    return { 'data': {'text/html' : div(html),
                      'text/plain' : txt },
             'metadata' : {} }