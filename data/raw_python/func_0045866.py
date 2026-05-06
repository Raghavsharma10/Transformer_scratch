def _get_item(self, question_id):
        """we need a middle-man method to convert the unique "assessment-session"
            authority question_ids into "real" itemIds
        BUT this also has to return the "magic" item, so we can't rely
        on
                question = self.get_question(question_id)
                ils = self._get_item_lookup_session()
                return ils.get_item(Id(question._my_map['itemId']))
        """
        question_map = self._get_question_map(question_id)  # Throws NotFound()
        real_question_id = Id(question_map['questionId'])
        return self._get_item_lookup_session().get_item(real_question_id)