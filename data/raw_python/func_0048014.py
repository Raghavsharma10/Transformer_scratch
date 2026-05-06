def delete_item(self, item_id):
        """Deletes the ``Item`` identified by the given ``Id``.

        arg:    item_id (osid.id.Id): the ``Id`` of the ``Item`` to
                delete
        raise:  NotFound - an ``Item`` was not found identified by the
                given ``Id``
        raise:  NullArgument - ``item_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(item_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        # This needs to be updated to actually check for AssessmentsTaken (and does that find even work?)
        if collection.find({'itemIds': str(item_id)}).count() != 0:
            raise errors.IllegalState('this Item is being used in one or more Assessments')
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        item_map = collection.find_one({'_id': ObjectId(item_id.get_identifier())})
        objects.Item(osid_object_map=item_map,
                     runtime=self._runtime,
                     proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(item_id.get_identifier())})