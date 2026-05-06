def query(self, query_criteria, valid_record=None):

        '''
            a core method for querying model valid data with criteria

            **NOTE: input is only returned if all fields & qualifiers are valid for model

            :param query_criteria: dictionary with model field names and query qualifiers
            :param valid_record: dictionary with model valid record
            :return: boolean (or QueryValidationError)

            an example of how to construct the query_criteria argument:

            query_criteria = {
                '.path.to.number': {
                    'min_value': 4.5
                },
                '.path.to.string': {
                    'must_contain': [ '\\regex' ]
                }
            }

            **NOTE: for a full list of operators for query_criteria based upon field
                    datatype, see either the query-rules.json file or REFERENCE file
        '''

        __name__ = '%s.query' % self.__class__.__name__
        _query_arg = '%s(query_criteria={...})' % __name__
        _record_arg = '%s(valid_record={...})' % __name__

    # validate input
        if not isinstance(query_criteria, dict):
            raise ModelValidationError('%s must be a dictionary.' % _query_arg)

    # convert javascript dot_path to class dot_path
        criteria_copy = {}
        equal_fields = []
        dot_fields = []
        for key, value in query_criteria.items():
            copy_key = key
            if not key:
                copy_key = '.'
            else:
                if key[0] != '.':
                    copy_key = '.%s' % key
                    dot_fields.append(copy_key)
            criteria_copy[copy_key] = value
            if value.__class__ in self._datatype_classes[0:4]:
                criteria_copy[copy_key] = {
                    'equal_to': value
                }
                equal_fields.append(copy_key)

    # validate query criteria against query rules
        query_kwargs = {
            'fields_dict': criteria_copy,
            'fields_rules': self.queryRules,
            'declared_value': False
        }
        try:
            self._validate_fields(**query_kwargs)
        except ModelValidationError as err:
            message = err.error['message']
            for field in equal_fields:
                equal_error = 'field %s qualifier equal_to' % field
                if message.find(equal_error) > -1:
                    message = message.replace(equal_error, 'field %s' % field)
                    break
            field_pattern = re.compile('ield\s(\..*?)\s')
            field_name = field_pattern.findall(message)
            if field_name:
                if field_name[0] in dot_fields:
                    def _replace_field(x):
                        return 'ield %s ' % x.group(1)[1:]
                    message = field_pattern.sub(_replace_field, message)
            raise QueryValidationError(message)

    # query test record
        if valid_record:
            if not isinstance(valid_record, dict):
                raise ModelValidationError('%s must be a dictionary.' % _record_arg)
            for key, value in criteria_copy.items():
                eval_outcome = self._evaluate_field(valid_record, key, value)
                if not eval_outcome:
                    return False

        return True