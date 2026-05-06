def nack(messageid, subscriptionid, transactionid=None):
    """STOMP negative acknowledge command.

    NACK is the opposite of ACK. It is used to tell the server that the client
    did not consume the message. The server can then either send the message to
    a different client, discard it, or put it in a dead letter queue. The exact
    behavior is server specific.

    messageid:
        This is the id of the message we are acknowledging,
        what else could it be? ;)

    subscriptionid:
        This is the id of the subscription that applies to the message.

    transactionid:
        This is the id that all actions in this transaction
        will have. If this is not given then a random UUID
        will be generated for this.

    """
    header = 'subscription:%s\nmessage-id:%s' % (subscriptionid, messageid)

    if transactionid:
        header += '\ntransaction:%s' % transactionid

    return "NACK\n%s\n\n\x00\n" % header