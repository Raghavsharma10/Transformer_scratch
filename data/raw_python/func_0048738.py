def selected(self, ids):
        """
        Return the selected options as a list of tuples
        """
        result = copy(self.choices)
        result = filter(lambda x: x[0] in ids, result)
        # result = ((item, item) for item in result)
        return list(result)