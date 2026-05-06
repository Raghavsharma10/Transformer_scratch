def call(self, callname, arguments=None):
        """Executed on each scheduled iteration"""
        # See if a method override exists
        action = getattr(self.api, callname, None)
        if action is None:
            try:
                action = self.api.ENDPOINT_OVERRIDES.get(callname, None)
            except AttributeError:
                action = callname

        if not callable(action):
            request = self._generate_request(action, arguments)
            if action is None:
                return self._generate_result(
                    callname, self.api.call(*call_args(callname, arguments)))
            return self._generate_result(
                callname, self.api.call(*call_args(action, arguments)))

        request = self._generate_request(callname, arguments)
        return self._generate_result(callname, action(request))