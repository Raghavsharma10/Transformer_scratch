def notify(self, value):
        """
        Increment or decrement the value, according to the given value's sign

        The value should be an integer, an attempt to cast it to integer will be made
        """
        value = int(value)

        with self.lock:
            self.value += value