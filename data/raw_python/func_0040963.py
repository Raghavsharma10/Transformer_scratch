def _notify_on_condition(self, test_message=None, **kwargs):
        """Returns the value of `notify_on_condition` or False.
        """
        if test_message:
            return True
        else:
            return self.enabled and self.notify_on_condition(**kwargs)