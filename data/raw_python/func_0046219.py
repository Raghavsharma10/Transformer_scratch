def set_review_whether_correct(self,
                                   during_attempt=None,
                                   after_attempt=None,
                                   before_deadline=None,
                                   after_deadline=None):
        """stub"""
        whether_correct = self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect']
        if during_attempt is not None:
            whether_correct['duringAttempt'] = bool(during_attempt)
        if after_attempt is not None:
            whether_correct['afterAttempt'] = bool(after_attempt)
        if before_deadline is not None:
            whether_correct['beforeDeadline'] = bool(before_deadline)
        if after_deadline is not None:
            whether_correct['afterDeadline'] = bool(after_deadline)