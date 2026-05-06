def description(self):
        """
        A list of the metrics this query will ask for.
        """

        if 'metrics' in self.raw:
            metrics = self.raw['metrics']
            head = metrics[0:-1] or metrics[0:1]
            text = ", ".join(head)
            if len(metrics) > 1:
                tail = metrics[-1]
                text = text + " and " + tail
        else:
            text = 'n/a'

        return text