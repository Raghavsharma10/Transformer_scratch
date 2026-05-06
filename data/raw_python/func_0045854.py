def _insert_part_map(self, part_map, index=-1):
        """ add a part map to self._my_map['assessmentParts']"""
        if index == -1:
            self._my_map['assessmentParts'].append(part_map)
        else:
            self._my_map['assessmentParts'].insert(index, part_map)