def get_assessment_parts_by_item(self, item_id):
        """Gets the assessment parts containing the given item.

        In plenary mode, the returned list contains all known assessment
        parts or an error results. Otherwise, the returned list may
        contain only those assessment parts that are accessible through
        this session.

        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.authoring.AssessmentPartList) - the
                returned ``AssessmentPart list``
        raise:  NotFound - ``item_id`` is not found
        raise:  NullArgument - ``item_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetCompositionSession.get_compositions_by_asset
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'itemIds': {'$in': [str(item_id)]}},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentPartList(result, runtime=self._runtime)