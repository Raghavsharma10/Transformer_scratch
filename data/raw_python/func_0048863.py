def get_sequence_rules_by_genus_type(self, sequence_rule_genus_type):
        """Gets a ``SequenceRuleList`` corresponding to the given sequence rule genus ``Type`` which does not include sequence rule of genus types derived from the specified ``Type``.

        arg:    sequence_rule_genus_type (osid.type.Type): a sequence
                rule genus type
        return: (osid.assessment.authoring.SequenceRuleList) - the
                returned ``SequenceRule`` list
        raise:  NullArgument - ``sequence_rule_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(sequence_rule_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.SequenceRuleList(result, runtime=self._runtime, proxy=self._proxy)