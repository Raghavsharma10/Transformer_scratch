def connection(self):
        """Snow connection instance, stores a `pysnow.Client` instance and `pysnow.Resource` instances

        Creates a new :class:`pysnow.Client` object if it doesn't exist in the app slice of the context stack

        :returns: :class:`pysnow.Client` object
        """

        ctx = stack.top.app
        if ctx is not None:
            if not hasattr(ctx, 'snow'):
                if self._client_type_oauth:
                    if not self._token_updater:
                        warnings.warn("No token updater has been set. Token refreshes will be ignored.")

                    client = self._get_oauth_client()
                else:
                    client = self._get_basic_client()

                if self._parameters:
                    # Set parameters passed on app init
                    client.parameters = self._parameters

                ctx.snow = client

            return ctx.snow