def _update_assessment_parts_map(self, part_list):
        """Updates the part map.

        Called before question list gets updated if it is determined that the
        sections assessmentPart map is out of date with the current part list.

        """
        for part in part_list:
            # perhaps look for a "level offset"?
            level = part._level_in_section  # plus or minus "level offset"?
            if str(part.get_id()) not in self._part_ids():
                self._insert_part_map(get_default_part_map(
                    part.get_id(), level, part.are_items_sequential()),
                    index=part_list.index(part))