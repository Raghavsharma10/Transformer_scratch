def get_next_assessment_part_id(self, assessment_part_id=None):
        """This supports the basic simple sequence case. Can be overriden in a record for other cases"""
        if assessment_part_id is None:
            part_id = self.get_id()
        else:
            part_id = assessment_part_id
        return get_next_part_id(part_id,
                                runtime=self._runtime,
                                proxy=self._proxy,
                                sequestered=True)[0]