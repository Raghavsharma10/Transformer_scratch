def create_handler_func(self, method):
        """
        create_handler_func
        """
        def _handler(callback, schema=None):
            """
            _handler
            """
            # reentrant default is False [POST, DELETE, PUT]
            reentrant = False
            if method == "get":
                reentrant = True

            self.handlers.append({
                "method": method,
                "callback": callback,
                "schema": schema,
                "reentrant": reentrant
            })
            return self

        return _handler