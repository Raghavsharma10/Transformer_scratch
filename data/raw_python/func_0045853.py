def _get_part_map(self, part_id):
        """ from self._my_map['assessmentParts'], return the one part map
        with ID that matches the one passed in"""
        return [p for p in self._my_map['assessmentParts']
                if p['assessmentPartId'] == str(part_id)][0]