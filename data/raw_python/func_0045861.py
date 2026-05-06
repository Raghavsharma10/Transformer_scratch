def _update_questions(self):
        """Updates questions known to this Section"""
        if self.is_simple_section():
            return  # we don't need to go through any this for simple sections
        # ideally, we would update the parts map and questions list
        # at the same time as _get_parts(), to not run into the
        # issue where magic parts are initialized (with items)
        # ignorant of their "sibling" magic part items...
        # because the section hasn't been updated or saved to database
        part_list = self._get_parts()
        if len(part_list) > len(self._my_map['assessmentParts']):
            self._update_assessment_parts_map(part_list)
            self._update_questions_list(part_list)
            self._save()