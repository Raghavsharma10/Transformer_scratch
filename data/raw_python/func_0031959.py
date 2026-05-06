def update(self, value):
        """
        Update the current rate with the given value.
        The value must be an integer.
        """

        value = int(value)

        with self.lock:
            self.value += value