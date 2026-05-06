def new_status(self, new_status):
        """
        Sets the new_status of this BuildSetStatusChangedEvent.

        :param new_status: The new_status of this BuildSetStatusChangedEvent.
        :type: str
        """
        allowed_values = ["NEW", "DONE", "REJECTED"]
        if new_status not in allowed_values:
            raise ValueError(
                "Invalid value for `new_status` ({0}), must be one of {1}"
                .format(new_status, allowed_values)
            )

        self._new_status = new_status