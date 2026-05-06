def get_feedback(self, question_id):
        """get feedback for item"""
        response = self.get_response(question_id)
        item = self._get_item(response.get_item_id())
        if response.is_answered():
            try:
                return item.get_feedback_for_response(response)
            except errors.IllegalState:
                pass
        else:
            return item.get_feedback()