def listen(zelf):
        """
        listen indefinitely, handling messages as they come

        all redis specific exceptions are handled, anything your handler raises
        will not be handled. setting active to False on the Listener object
        will gracefully stop the listen() function
        """
        while zelf.active:
            try:
                msg = zelf.r.master.brpoplpush(zelf.lijst, zelf._processing,
                                               zelf.read_time)
                if msg:
                    # NOTE(tr3buchet): got a message, process it
                    LOG.debug('received: |%s|' % msg)
                    zelf._call_handler(msg)
            except zelf.r.generic_error:
                LOG.exception('')
            finally:
                time.sleep(0)