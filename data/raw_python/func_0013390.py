def _findLocation(self, reference_name, start, end):
        """
        return a location key form the locationMap
        """
        try:
            # TODO - sequence_annotations does not have build?
            return self._locationMap['hg19'][reference_name][start][end]
        except:
            return None