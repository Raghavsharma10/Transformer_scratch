def _simple_dispatch(self, name, params):
        """
        Dispatch method
        """
        # Normalize parameters
        if params:
            if isinstance(params, (list, tuple)):
                params = [jabsorb.from_jabsorb(param) for param in params]
            else:
                params = {key: jabsorb.from_jabsorb(value)
                          for key, value in params.items()}

        # Dispatch like JSON-RPC
        return super(JabsorbRpcDispatcher, self)._simple_dispatch(name, params)