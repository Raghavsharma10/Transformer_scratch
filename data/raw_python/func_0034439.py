def auth(self, password):
        """Request for authentication in a password-protected Redis server.
        Redis can be instructed to require a password before allowing clients
        to execute commands. This is done using the ``requirepass`` directive
        in the configuration file.

        If the password does not match, an
        :exc:`~tredis.exceptions.AuthError` exception
        will be raised.

        :param password: The password to authenticate with
        :type password: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.AuthError`,
                 :exc:`~tredis.exceptions.RedisError`

        """
        future = concurrent.TracebackFuture()

        def on_response(response):
            """Process the redis response

            :param response: The future with the response
            :type response: tornado.concurrent.Future

            """
            exc = response.exception()
            if exc:
                if exc.args[0] == b'invalid password':
                    future.set_exception(exceptions.AuthError(exc))
                else:
                    future.set_exception(exc)
            else:
                future.set_result(response.result())

        execute_future = self._execute([b'AUTH', password], b'OK')
        self.io_loop.add_future(execute_future, on_response)
        return future