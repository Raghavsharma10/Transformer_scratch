def get_sequence_rule_form_for_create(self, assessment_part_id, next_assessment_part_id, sequence_rule_record_types):
        """Gets the sequence rule form for creating new sequence rules between two assessment parts.

        A new form should be requested for each create transaction.

        arg:    assessment_part_id (osid.id.Id): the source assessment
                part ``Id``
        arg:    next_assessment_part_id (osid.id.Id): the target
                assessment part ``Id``
        arg:    sequence_rule_record_types (osid.type.Type[]): array of
                sequence rule record types
        return: (osid.assessment.authoring.SequenceRuleForm) - the
                sequence rule form
        raise:  InvalidArgument - ``assessment_part_id`` and
                ``next_assessment_part_id`` not on the same assessment
        raise:  NotFound - ``assessment_part_id`` or
                ``next_assessment_part_id`` is not found
        raise:  NullArgument - ``assessment_part_id,
                next_assessment_part_id`` , or
                ``sequence_rule_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        for arg in sequence_rule_record_types:
            if not isinstance(arg, ABCId):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID ${arg0_type}')
        if sequence_rule_record_types == []:
            obj_form = objects.SequenceRuleForm(
                bank_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy,
                next_assessment_part_id=next_assessment_part_id,
                assessment_part_id=assessment_part_id)
        else:
            obj_form = objects.SequenceRuleForm(
                bank_id=self._catalog_id,
                record_types=sequence_rule_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy,
                next_assessment_part_id=next_assessment_part_id,
                assessment_part_id=assessment_part_id)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form