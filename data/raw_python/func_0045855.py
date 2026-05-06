def _load_simple_section_questions(self, item_ids):
        """For loading the simple section case (common)

        just load the questions for the section, and insert the one part
        into assessment part map.

        """
        self._insert_part_map(
            get_default_part_map(self._assessment_part_id,
                                 0,
                                 self._assessment_part.are_items_sequential()))
        lookup_session = self._get_item_lookup_session()
        items = lookup_session.get_items_by_ids(item_ids)
        display_num = 1
        for item in items:
            question_id = item.get_question().get_id()
            self._my_map['questions'].append(get_default_question_map(
                item.get_id(),
                question_id,
                self._assessment_part_id,
                [display_num]))
            display_num += 1
        self._save()