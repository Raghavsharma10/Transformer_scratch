def __handler(self, connection, event):
        """
        Handles an IRC event
        """
        try:
            # Find local handler
            method = getattr(self, "on_{0}".format(event.type))
        except AttributeError:
            pass
        else:
            try:
                # Call it
                return method(connection, event)
            except Exception as ex:
                logging.exception("Error calling handler: %s", ex)