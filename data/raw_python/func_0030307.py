def old_status(self, old_status):
        """
        Sets the old_status of this BuildSetStatusChangedEvent.

        :param old_status: The old_status of this BuildSetStatusChangedEvent.
        :type: str
        """
        allowed_values = ["NEW", "DONE", "REJECTED"]
        if old_status not in allowed_values:
            raise ValueError(
                "Invalid value for `old_status` ({0}), must be one of {1}"
                .format(old_status, allowed_values)
            )

        self._old_status = old_status