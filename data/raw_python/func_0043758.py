def _evaluate_field(self, record_dict, field_name, field_criteria):

        ''' a helper method for evaluating record values based upon query criteria

        :param record_dict: dictionary with model valid data to evaluate
        :param field_name: string with path to root of query field
        :param field_criteria: dictionary with query operators and qualifiers
        :return: boolean (True if no field_criteria evaluate to false)
        '''

    # determine value existence criteria
        value_exists = True
        if 'value_exists' in field_criteria.keys():
            if not field_criteria['value_exists']:
                value_exists = False

    # validate existence of field
        field_exists = True
        try:
            record_values = self._walk(field_name, record_dict)
        except:
            field_exists = False

    # evaluate existence query criteria
        if value_exists != field_exists:
            return False
        elif not value_exists:
            return True

    # convert javascript dot_path to class dot_path
        field_key = field_name
        if not field_name:
            field_key = '.'
        else:
            if field_name[0] != '.':
                field_key = '.%s' % field_name

    # evaluate other query criteria
        for key, value in field_criteria.items():
            if key in ('min_size', 'min_length'):
                found = False
                if self.keyMap[field_key]['value_datatype'] == 'map':
                    for record_value in record_values:
                        record_size = sys.getsizeof(json.dumps(str(record_value)).replace(' ','')) - 51
                        if record_size >= value:
                            found = True
                            break
                else:
                    for record_value in record_values:
                        if len(record_value) >= value:
                            found = True
                            break
                if not found:
                    return False
            elif key in ('max_size', 'max_length'):
                found = False
                if self.keyMap[field_key]['value_datatype'] == 'map':
                    for record_value in record_values:
                        record_size = sys.getsizeof(json.dumps(str(record_value)).replace(' ','')) - 51
                        if record_size <= value:
                            found = True
                            break
                else:
                    for record_value in record_values:
                        if len(record_value) <= value:
                            found = True
                            break
                if not found:
                    return False
            elif key == 'min_value':
                found = False
                for record_value in record_values:
                    if record_value >= value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'max_value':
                found = False
                for record_value in record_values:
                    if record_value <= value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'equal_to':
                found = False
                for record_value in record_values:
                    if record_value == value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'greater_than':
                found = False
                for record_value in record_values:
                    if record_value > value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'less_than':
                found = False
                for record_value in record_values:
                    if record_value < value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'excluded_values':
                for record_value in record_values:
                    if record_value in value:
                        return False
            elif key == 'discrete_values':
                found = False
                for record_value in record_values:
                    if record_value in value:
                        found = True
                        break
                if not found:
                    return False
            elif key == 'integer_data':
                found = False
                dummy_int = 1
                for record_value in record_values:
                    if record_value.__class__ == dummy_int.__class__:
                        found = True
                        break
                if value != found:
                    return False
            elif key == 'byte_data':
                found = False
                for record_value in record_values:
                    try:
                        decoded_bytes = b64decode(record_value)
                    except:
                        decoded_bytes = ''
                    if isinstance(decoded_bytes, bytes):
                        found = True
                        break
                if value != found:
                    return False
            elif key == 'must_contain':
                for regex in value:
                    regex_pattern = re.compile(regex)
                    found = False
                    for record_value in record_values:
                        if regex_pattern.findall(record_value):
                            found = True
                            break
                    if not found:
                        return False
            elif key == 'must_not_contain':
                for regex in value:
                    regex_pattern = re.compile(regex)
                    found = False
                    for record_value in record_values:
                        if regex_pattern.findall(record_value):
                            found = True
                            break
                    if found:
                        return False
            elif key == 'contains_either':
                found = False
                for regex in value:
                    regex_pattern = re.compile(regex)
                    for record_value in record_values:
                        if regex_pattern.findall(record_value):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False
            elif key == 'unique_values':
                for record_value in record_values:
                    unique_values = True
                    if len(record_value) != len(set(record_value)):
                        unique_values = False
                    if value != unique_values:
                        return False

        return True