def get_score_system_id(self):
        """Gets a score system ``Id`` for the assessment.

        return: (osid.id.Id) - the grade system
        raise:  IllegalState - ``is_score()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['scoreSystemId']):
            raise errors.IllegalState('this AssessmentTaken has no score_system')
        mgr = self._get_provider_manager('ID')
        if not mgr.supports_id_lookup():
            raise errors.OperationFailed('Id does not support Id lookup')
        lookup_session = mgr.get_id_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_no_catalog_view()
        osid_object = lookup_session.get_id(self.get_score_system_id())
        return osid_object