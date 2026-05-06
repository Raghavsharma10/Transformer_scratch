def get_my_item_id_from_section(self, section):
        """returns the first item associated with this magic Part Id in the Section"""
        for question_map in section._my_map['questions']:
            if question_map['assessmentPartId'] == str(self.get_id()):
                return section.get_question(question_map=question_map).get_id()
        raise IllegalState('This Part currently has no Item in the Section')