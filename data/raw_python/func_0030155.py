def value_labels(self):
        """Return a map of column code values mapped to labels, for columns that have a label column

        If the column is not assocaited with a label column, it returns an identity map.

        WARNING! This reads the whole partition, so it is really slow

        """

        from operator import itemgetter

        card = self.pstats.nuniques

        if self.label:
            ig = itemgetter(self.name, self.label.name)
        elif self.pstats.nuniques < MAX_LABELS:
            ig = itemgetter(self.name, self.name)
        else:
            return {}

        label_set = set()
        for row in self._partition:
            label_set.add(ig(row))

            if len(label_set) >= card:
                break

        d = dict(label_set)

        assert len(d) == len(label_set)  # Else the label set has multiple values per key

        return d