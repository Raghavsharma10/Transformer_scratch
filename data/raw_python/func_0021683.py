def _do_post_request_tasks(self, response_data):
        """Handle actions that need to be done with every response

        I'm not sure what these session_ops are actually used for yet, seems to
        be a way to tell the client to do *something* if needed.
        """
        try:
            sess_ops = response_data.get('ops', [])
        except AttributeError:
            pass
        else:
            self._session_ops.extend(sess_ops)