def set_review_solution(self,
                            during_attempt=None,
                            after_attempt=None,
                            before_deadline=None,
                            after_deadline=None):
        """stub"""
        solution = self.my_osid_object_form._my_map['reviewOptions']['solution']
        if during_attempt is not None:
            solution['duringAttempt'] = bool(during_attempt)
        if after_attempt is not None:
            solution['afterAttempt'] = bool(after_attempt)
        if before_deadline is not None:
            solution['beforeDeadline'] = bool(before_deadline)
        if after_deadline is not None:
            solution['afterDeadline'] = bool(after_deadline)