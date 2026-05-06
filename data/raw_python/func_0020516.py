def get_name_and_value(self, label_type):
        """
        Return tuple of (label name, label value)
        Raises KeyError if label doesn't exist
        """
        if label_type in self._label_values:
            return self._label_values[label_type]
        else:
            return (label_type, self._df_labels[label_type])