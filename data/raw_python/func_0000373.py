def is_answer_valid(self, ans):
        """Validate user's answer against available choices."""
        return ans in [str(i+1) for i in range(len(self.choices))]