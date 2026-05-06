def _diff_attr(self, attr_name, filename):
        """
        Return the difference between
        `self.cobertura2.<attr_name>(filename)` and
        `self.cobertura1.<attr_name>(filename)`.

        This generic method is meant to diff the count of methods that return
        counts for a given file `filename`, e.g. `Cobertura.total_statements`,
        `Cobertura.total_misses`, ...

        The returned count may be a float.
        """
        if filename is not None:
            files = [filename]
        else:
            files = self.files()

        total_count = 0.0
        for filename in files:
            if self.cobertura1.has_file(filename):
                method = getattr(self.cobertura1, attr_name)
                count1 = method(filename)
            else:
                count1 = 0.0
            method = getattr(self.cobertura2, attr_name)
            count2 = method(filename)
            total_count += count2 - count1

        return total_count