def is_feedback_available(self, question_id):
        """is feedback available for item"""
        response = self.get_response(question_id)
        item = self._get_item(question_id)
        if response.is_answered():
            return item.is_feedback_available_for_response(response)
        return item.is_feedback_available()