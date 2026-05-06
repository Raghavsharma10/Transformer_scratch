def handle(self, event):
        """Decorator for adding a handler function for a particular event.

        Usage:

            my_client = Client()

            @my_client.handle("WELCOME")
            def welcome_handler(client, *params):
                # Do something with the event.
                pass
        """
        def dec(func):
            self.add_handler(event, func)
            return func
        return dec