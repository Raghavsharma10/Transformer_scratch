def by_filenumber(self):
        """
        Iterates over categories and returns a filtered datamat.

        If a categories object is attached, the images object for the given
        category is returned as well (else None is returned).

        Returns:
            (datamat, categories) : A tuple that contains first the filtered
                datamat (has only one category) and second the associated
                categories object (if it is available, None otherwise)
        """
        for value in np.unique(self.filenumber):
            file_fm = self.filter(self.filenumber == value)
            if self._categories:
                yield (file_fm, self._categories[self.category[0]][value])
            else:
                yield (file_fm, None)