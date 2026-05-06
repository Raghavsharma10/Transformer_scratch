def get_assessment_part_items(self, assessment_part_id):
        """Gets the list of items mapped to the given ``AssessmentPart``.

        In plenary mode, the returned list contains all known items or
        an error results. Otherwise, the returned list may contain only
        those items that are accessible through this session.

        arg:    assessment_part_id (osid.id.Id): ``Id`` of the
                ``AssessmentPart``
        return: (osid.assessment.ItemList) - list of items
        raise:  NotFound - ``assessment_part_id`` not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
        lookup_session = mgr.get_assessment_part_lookup_session(proxy=self._proxy)
        if self._catalog_view == ISOLATED:
            lookup_session.use_isolated_bank_view()
        else:
            lookup_session.use_federated_bank_view()
        item_ids = lookup_session.get_assessment_part(assessment_part_id).get_item_ids()
        mgr = self._get_provider_manager('ASSESSMENT')
        lookup_session = mgr.get_item_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        return lookup_session.get_items_by_ids(item_ids)