def get_correctness(self, question_id):
        """get measure of correctness for the question"""
        response = self.get_response(question_id)
        if response.is_answered():
            item = self._get_item(response.get_item_id())
            return item.get_correctness_for_response(response)
        raise errors.IllegalState()