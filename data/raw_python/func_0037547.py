def _func_router(self, msg, fname, **config):
        """
        This method routes the messages based on the params and calls the
        appropriate method to process the message. The utility of the method is
        to cope up with the major message change during different releases.
        """
        FNAME = 'handle_%s_autocloud_%s'

        if ('compose_id' in msg['msg'] or 'compose_job_id' in msg['msg'] or
                'autocloud.compose' in msg['topic']):
            return getattr(self, FNAME % ('v2', fname))(msg, **config)
        else:
            return getattr(self, FNAME % ('v1', fname))(msg, **config)