def get_level(self):
        """Gets the ``Grade`` corresponding to the assessment difficulty.

        return: (osid.grading.Grade) - the level
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['levelId']):
            raise errors.IllegalState('this Assessment has no level')
        mgr = self._get_provider_manager('GRADING')
        if not mgr.supports_grade_lookup():
            raise errors.OperationFailed('Grading does not support Grade lookup')
        lookup_session = mgr.get_grade_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_gradebook_view()
        osid_object = lookup_session.get_grade(self.get_level_id())
        return osid_object