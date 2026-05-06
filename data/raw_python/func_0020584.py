def _send(self, javascript):
        """
        Establishes a socket connection to the zombie.js server and sends
        Javascript instructions.

        :param js: the Javascript string to execute
        """

        # Prepend JS to switch to the proper client context.
        message = """
            var _ctx = ctx_switch('%s'),
                browser = _ctx[0],
                ELEMENTS = _ctx[1];
            %s
        """ % (id(self), javascript)

        response = self.connection.send(message)

        return self._handle_response(response)