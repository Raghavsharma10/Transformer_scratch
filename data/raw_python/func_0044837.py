def get_message(zelf):
        """
        get one message if available else return None
        if message is available returns the result of handler(message)
        does not block!

        if you would like to call your handler manually, this is the way to
        go. don't pass in a handler to Listener() and the default handler will
        log and return the message for your own manual processing
        """
        try:
            message = zelf.r.master.rpoplpush(zelf.lijst, zelf._processing)
            if message:
                # NOTE(tr3buchet): got a message, process it
                LOG.debug('received: |%s|' % message)
                return zelf._call_handler(message)
        except zelf.r.generic_error:
            LOG.exception('')