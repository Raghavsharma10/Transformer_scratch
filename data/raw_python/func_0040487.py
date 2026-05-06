def csv(self):
        """Parse raw response as csv and return row object list.
        """
        lines = self._parsecsv(self.raw)

        # set keys from header line (first line)
        keys = next(lines)

        for line in lines:
            yield dict(zip(keys, line))