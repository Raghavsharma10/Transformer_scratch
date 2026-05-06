def data_msg( msg, mtype=None ):
    """
    Return a Jupyter display_data message, in both HTML & text formats, by 
    formatting a given single message. The passed message may be:
      * An exception (including a KrnlException): will generate an error message
      * A list of messages (with \c mtype equal to \c multi)
      * A single message

      @param msg (str,list): a string, or a list of format string + args,
        or an iterable of (msg,mtype)
      @param mtype (str): the message type (used for the CSS class). If
        it's \c multi, then \c msg will be treated as a multi-message. If
        not passed, \c krn-error will be used for exceptions and \c msg for
        everything else
    """
    if isinstance(msg,KrnlException):
        return msg()    # a KrnlException knows how to format itself
    elif isinstance(msg,Exception):
        return KrnlException(msg)()
    elif mtype == 'multi':
        return data_msglist( msg )
    else:
        return data_msglist( [ (msg, mtype) ] )