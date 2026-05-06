def get_next_assessment_part_id(self, assessment_part_id):
        """This supports the basic simple sequence case. Can be overriden in a record for other cases"""
        if self.has_next_assessment_part(assessment_part_id):
            return Id(self._my_map['childIds'][self._my_map['childIds'].index(str(assessment_part_id)) + 1])