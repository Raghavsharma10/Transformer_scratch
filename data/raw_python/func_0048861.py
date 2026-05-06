def get_sequence_rule(self, sequence_rule_id):
        """Gets the ``SequenceRule`` specified by its ``Id``.

        arg:    sequence_rule_id (osid.id.Id): ``Id`` of the
                ``SequenceRule``
        return: (osid.assessment.authoring.SequenceRule) - the sequence
                rule
        raise:  NotFound - ``sequence_rule_id`` not found
        raise:  NullArgument - ``sequence_rule_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(sequence_rule_id, 'assessment_authoring').get_identifier())},
                 **self._view_filter()))
        return objects.SequenceRule(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)