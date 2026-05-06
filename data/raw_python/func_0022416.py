def wait_for_exists(self, timeout=0, *args, **selectors):
        """
        Wait for the object which has *selectors* within the given timeout.

        Return true if the object *appear* in the given timeout. Else return false.
        """
        return self.device(**selectors).wait.exists(timeout=timeout)