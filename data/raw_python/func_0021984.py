def _maybe_registered(self, failure, new_reg):
        """
        If the registration already exists, we should just load it.
        """
        failure.trap(ServerError)
        response = failure.value.response
        if response.code == http.CONFLICT:
            reg = new_reg.update(
                resource=messages.UpdateRegistration.resource_type)
            uri = self._maybe_location(response)
            return self.update_registration(reg, uri=uri)
        return failure