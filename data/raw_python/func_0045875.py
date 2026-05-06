def get_confused_learning_objective_ids(self, question_id):
        """get confused objective ids available for the question"""
        response = self.get_response(question_id)
        if response.is_answered():
            item = self._get_item(response.get_item_id())
            return item.get_confused_learning_objective_ids_for_response(response)
        raise errors.IllegalState()