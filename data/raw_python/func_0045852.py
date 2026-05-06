def _initialize_part_map(self):
        """Sets up assessmentPartMap with as much information as is initially available."""
        self._my_map['assessmentParts'] = []
        self._my_map['questions'] = []
        item_ids = self._assessment_part.get_item_ids()
        if item_ids.available():
            # This is a simple section:
            self._load_simple_section_questions(item_ids)
        else:
            # This goes down the winding path...
            # let's not call this...seems redundant, and per Jeff, this might
            # save us time.
            # self._update_questions()
            pass