def delete_sequence_rule(self, sequence_rule_id):
        """Deletes a ``SequenceRule``.

        arg:    sequence_rule_id (osid.id.Id): the ``Id`` of the
                ``SequenceRule`` to remove
        raise:  NotFound - ``sequence_rule_id`` not found
        raise:  NullArgument - ``sequence_rule_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        if not isinstance(sequence_rule_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        sequence_rule_map = collection.find_one(
            dict({'_id': ObjectId(sequence_rule_id.get_identifier())},
                 **self._view_filter()))

        objects.SequenceRule(osid_object_map=sequence_rule_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(sequence_rule_id.get_identifier())})