def add_delegate(self, callback):
        """ Registers a new delegate callback

            The prototype should be function(data), where data will be the decoded json push

            Args:
                callback (function): method to trigger when push center receives events
        """

        if callback in self._delegate_methods:
            return

        self._delegate_methods.append(callback)