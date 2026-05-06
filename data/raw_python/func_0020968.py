def send(dest, msg, transactionid=None):
    """STOMP send command.

    dest:
        This is the channel we wish to subscribe to

    msg:
        This is the message body to be sent.

    transactionid:
        This is an optional field and is not needed
        by default.

    """
    transheader = ''

    if transactionid:
        transheader = 'transaction: %s\n' % transactionid

    return "SEND\ndestination: %s\n%s\n%s\x00\n" % (dest, transheader, msg)