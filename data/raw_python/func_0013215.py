def get_all(self):
        """ Returns a list populated by tuples of 2 elements, first one is
            a dict with all the labels and the second elemnt is the value
            of the metric itself
        """
        with mutex:
            items = self.values.items()

        result = []
        for k, v in items:
            # Check if is a single value dict (custom empty key)
            if not k or k == MetricDict.EMPTY_KEY:
                key = None
            else:
                key = decoder.decode(k)
            result.append((key, self.get(k)))

        return result