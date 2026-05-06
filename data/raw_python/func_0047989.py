def get_response_form(self, assessment_section_id, item_id):
        """Gets the response form for submitting an answer.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.AnswerForm) - an answer form
        raise:  IllegalState - ``has_assessment_section_begun()`` is
                ``false or is_assessment_section_over()`` is ``true``
        raise:  NotFound - ``assessment_section_id`` or ``item_id`` is
                not found, or ``item_id`` not part of
                ``assessment_section_id``
        raise:  NullArgument - ``assessment_section_id`` or ``item_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(item_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')

        # This is a little hack to get the answer record types from the Item's
        # first Answer record types. Should really get it from item genus types somehow:
        record_type_data_sets = get_registry('ANSWER_RECORD_TYPES', self._runtime)
        section = self.get_assessment_section(assessment_section_id)
        # because we're now giving session-unique question IDs
        question = section.get_question(item_id)
        ils = section._get_item_lookup_session()
        real_item_id = Id(question._my_map['itemId'])
        item = ils.get_item(real_item_id)
        item_map = item.object_map
        all_answers = item_map['answers']
        try:
            all_answers += [wa.object_map for wa in item.get_wrong_answers()]
        except AttributeError:
            pass

        answer_record_types = []
        if len(all_answers) > 0:
            for record_type_idstr in all_answers[0]['recordTypeIds']:
                identifier = Id(record_type_idstr).get_identifier()
                if identifier in record_type_data_sets:
                    answer_record_types.append(Type(**record_type_data_sets[identifier]))
        else:
            for record_type_idstr in item_map['question']['recordTypeIds']:
                identifier = Id(record_type_idstr).get_identifier()
                if identifier in record_type_data_sets:
                    answer_record_types.append(Type(**record_type_data_sets[identifier]))
        # Thus endith the hack.

        obj_form = objects.AnswerForm(
            bank_id=self._catalog_id,
            record_types=answer_record_types,
            item_id=item_id,
            catalog_id=self._catalog_id,
            assessment_section_id=assessment_section_id,
            runtime=self._runtime,
            proxy=self._proxy)
        obj_form._for_update = False  # This may be redundant
        self._forms[obj_form.get_id().get_identifier()] = not SUBMITTED
        return obj_form