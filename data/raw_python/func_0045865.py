def get_item_ids_for_assessment_part(self, assessment_part_id):
        """convenience method returns item ids associated with an assessment_part_id"""
        item_ids = []
        for question_map in self._my_map['questions']:
            if question_map['assessmentPartId'] == str(assessment_part_id):
                item_ids.append(Id(question_map['itemId']))
        return item_ids