def ack(messageid, transactionid=None):
    """STOMP acknowledge command.

    Acknowledge receipt of a specific message from the server.

    messageid:
        This is the id of the message we are acknowledging,
        what else could it be? ;)

    transactionid:
        This is the id that all actions in this transaction
        will have. If this is not given then a random UUID
        will be generated for this.

    """
    header = 'message-id: %s' % messageid

    if transactionid:
        header = 'message-id: %s\ntransaction: %s' % (messageid, transactionid)

    return "ACK\n%s\n\n\x00\n" % header