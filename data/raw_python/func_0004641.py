def get(self, timeout=None):
        """Retrieve results from all the output tubes."""
        valid = False
        result = None
        for tube in self._output_tubes:
            if timeout:
                valid, result = tube.get(timeout)
                if valid:
                    result = result[0]
            else:
                result = tube.get()[0]
        if timeout:
            return valid, result
        return result