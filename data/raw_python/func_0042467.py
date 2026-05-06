def _set_m2ms(self, old_m2ms):
        """
        Creates the same m2m relationships that the old
        object had.
        """

        for k, v in old_m2ms.items():
            if v:
                setattr(self, k, v)