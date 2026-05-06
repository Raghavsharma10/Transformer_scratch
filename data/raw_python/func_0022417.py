def wait_until_gone(self, timeout=0, *args, **selectors):
        """
        Wait for the object which has *selectors* within the given timeout.

        Return true if the object *disappear* in the given timeout. Else return false.
        """
        return self.device(**selectors).wait.gone(timeout=timeout)