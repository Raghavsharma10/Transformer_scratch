def is_correct(self, question_id):
        """is the question answered correctly"""
        response = self.get_response(question_id=question_id)
        if response.is_answered():
            item = self._get_item(response.get_item_id())
            return item.is_response_correct(response)
        raise errors.IllegalState()