def _init_map(self):
        """stub"""
        self.my_osid_object_form._my_map['reviewOptions'] = \
            dict(self._review_options_metadata['default_object_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect'] = \
            dict(self._whether_correct_metadata['default_object_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect']['duringAttempt'] = \
            bool(self._during_attempt_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect']['afterAttempt'] = \
            bool(self._after_attempt_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect']['beforeDeadline'] = \
            bool(self._before_deadline_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['whetherCorrect']['afterDeadline'] = \
            bool(self._after_deadline_metadata['default_boolean_values'][0])

        self.my_osid_object_form._my_map['reviewOptions']['solution'] = \
            dict(self._solutions_metadata['default_object_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['solution']['duringAttempt'] = False
        self.my_osid_object_form._my_map['reviewOptions']['solution']['afterAttempt'] = \
            bool(self._after_attempt_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['solution']['beforeDeadline'] = \
            bool(self._before_deadline_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['reviewOptions']['solution']['afterDeadline'] = \
            bool(self._after_deadline_metadata['default_boolean_values'][0])

        self.my_osid_object_form._my_map['maxAttempts'] = \
            list(self._max_attempts_metadata['default_integer_values'])[0]