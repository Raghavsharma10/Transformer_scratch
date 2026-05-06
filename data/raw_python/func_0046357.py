def has_next_assessment_part(self, assessment_part_id):
        """This supports the basic simple sequence case. Can be overriden in a record for other cases"""
        if not self.supports_child_ordering or not self.supports_simple_child_sequencing:
            raise AttributeError()  # Only available through a record extension
        if 'childIds' in self._my_map and str(assessment_part_id) in self._my_map['childIds']:
            if self._my_map['childIds'][-1] != str(assessment_part_id):
                return True
            else:
                return False
        raise errors.NotFound('the Part with Id ' + str(assessment_part_id) + ' is not a child of this Part')