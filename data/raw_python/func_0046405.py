def get_score_system(self):
        """Gets a grade system for the score.

        return: (osid.grading.GradeSystem) - the grade system
        raise:  IllegalState - ``is_scored()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['scoreSystemId']):
            raise errors.IllegalState('this AssessmentTaken has no score_system')
        mgr = self._get_provider_manager('GRADING')
        if not mgr.supports_grade_system_lookup():
            raise errors.OperationFailed('Grading does not support GradeSystem lookup')
        lookup_session = mgr.get_grade_system_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_gradebook_view()
        osid_object = lookup_session.get_grade_system(self.get_score_system_id())
        return osid_object