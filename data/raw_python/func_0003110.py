def _send(self, data, msg_type='ok', silent=False):
        """
        Send a response to the frontend and return an execute message
         @param data: response to send
         @param msg_type (str): message type: 'ok', 'raw', 'error', 'multi'
         @param silent (bool): suppress output
         @return (dict): the return value for the kernel
        """
        # Data to send back
        if data is not None:
            # log the message
            try:
                self._klog.debug(u"msg to frontend (%d): %.160s...", silent, data)
            except Exception as e:
                self._klog.warn(u"can't log response: %s", e)
            # send it to the frontend
            if not silent:
                if msg_type != 'raw':
                    data = data_msg(data, mtype=msg_type)
                self.send_response(self.iopub_socket, 'display_data', data)

        # Result message
        return {'status': 'error' if msg_type == 'error' else 'ok',
                # The base class will increment the execution count
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {}
                }