def set_value(self, labels, value):
        """ Sets a value in the container"""

        if labels:
            self._label_names_correct(labels)

        with mutex:
            self.values[labels] = value