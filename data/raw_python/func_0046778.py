def get_scaffold_objective_ids(self):
        """Assumes that a scaffold objective id is available"""
        section = self._assessment_section
        item_id = self.get_my_item_id_from_section(section)
        return section.get_confused_learning_objective_ids(item_id)