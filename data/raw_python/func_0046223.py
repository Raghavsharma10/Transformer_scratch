def can_review_whether_correct(self):
        """stub"""
        ao = self.my_osid_object.get_assessment_offered()
        attempt_complete = self.my_osid_object.has_ended()
        if ao.can_review_whether_correct_during_attempt() and not attempt_complete:
            return True
        if ao.can_review_whether_correct_after_attempt and attempt_complete:
            return True
        return False