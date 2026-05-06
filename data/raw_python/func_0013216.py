def add(self, labels, value):
        """ Add adds the given value to the Gauge. (The value can be
            negative, resulting in a decrease of the Gauge.)
        """

        try:
            current = self.get_value(labels)
        except KeyError:
            current = 0

        self.set_value(labels, current + value)