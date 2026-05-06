def get_name(self, label_type):
        """
        returns the most preferred label name
        if there isn't any correct name in the list
        it will return newest label name
        """
        if label_type in self._label_values:
            return self._label_values[label_type][0]
        else:
            return Labels.LABEL_NAMES[label_type][0]