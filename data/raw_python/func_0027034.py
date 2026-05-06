def remove_delegate(self, callback):
        """ Unregisters a registered delegate function or a method.

            Args:
                callback(function): method to trigger when push center receives events
        """

        if callback not in self._delegate_methods:
            return

        self._delegate_methods.remove(callback)