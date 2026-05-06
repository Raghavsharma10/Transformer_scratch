def bindToEndPoint(self,bindingEndpoint):
        """
        2-way binds the target endpoint to all other registered endpoints.
        """
        self.bindings[bindingEndpoint.instanceId] = bindingEndpoint
        bindingEndpoint.valueChangedSignal.connect(self._updateEndpoints)