def is_simple_section(self):
        """Tests if this section is simple (ie, items assigned directly to Section Part)."""
        item_ids = self._get_assessment_part(self._assessment_part_id).get_item_ids()
        if item_ids.available():
            return True
        return False