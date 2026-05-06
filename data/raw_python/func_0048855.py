def add_item(self, item_id, assessment_part_id):
        """Appends an item to an assessment part.

        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        arg:    assessment_part_id (osid.id.Id): ``Id`` of the
                ``AssessmentPart``
        raise:  AlreadyExists - ``item_id`` already part of
                ``assessment_part_id``
        raise:  NotFound - ``item_id`` or ``assessment_part_id`` not
                found
        raise:  NullArgument - ``item_id`` or ``assessment_part_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        # The item found check may want to be run through _get_provider_manager
        # so as to ensure access control:
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        if not isinstance(item_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        if (not isinstance(assessment_part_id, ABCId) and
                assessment_part_id.get_identifier_namespace() != 'assessment_authoring.AssessmentPart'):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        if item_id.get_identifier_namespace() != 'assessment.Item':
            if item_id.get_authority() != self._authority:
                raise errors.InvalidArgument()
            else:
                mgr = self._get_provider_manager('ASSESSMENT')
                admin_session = mgr.get_item_admin_session_for_bank(self._catalog_id, proxy=self._proxy)
                item_id = admin_session._get_item_id_with_enclosure(item_id)
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        item = collection.find_one({'_id': ObjectId(item_id.get_identifier())})
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        assessment_part = collection.find_one({'_id': ObjectId(assessment_part_id.get_identifier())})
        if 'itemIds' in assessment_part:
            if str(item_id) not in assessment_part['itemIds']:
                assessment_part['itemIds'].append(str(item_id))
        else:
            assessment_part['itemIds'] = [str(item_id)]
        collection.save(assessment_part)