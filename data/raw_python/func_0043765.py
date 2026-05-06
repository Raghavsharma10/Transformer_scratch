def _ingest_list(self, input_list, schema_list, path_to_root):

        '''
            a helper method for ingesting items in a list

        :return: valid_list
        '''

        valid_list = []

    # construct max list size
        max_size = None
        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
        if 'max_size' in self.keyMap[rules_path_to_root].keys():
            if not self.keyMap[rules_path_to_root]['max_size']:
                return valid_list
            else:
                max_size = self.keyMap[rules_path_to_root]['max_size']

    # iterate over items in input list
        if input_list:
            rules_index = self._datatype_classes.index(schema_list[0].__class__)
            rules_type = self._datatype_names[rules_index]
            for i in range(len(input_list)):
                item_path = '%s[%s]' % (path_to_root, i)
                value_match = False
                try:
                    item_index = self._datatype_classes.index(input_list[i].__class__)
                    item_type = self._datatype_names[item_index]
                    if item_type == rules_type:
                        value_match = True
                except:
                    value_match = False
                if value_match:
                    try:
                        if item_type == 'boolean':
                            valid_list.append(self._validate_boolean(input_list[i], item_path))
                        elif item_type == 'number':
                            valid_list.append(self._validate_number(input_list[i], item_path))
                        elif item_type == 'string':
                            valid_list.append(self._validate_string(input_list[i], item_path))
                        elif item_type == 'map':
                            valid_list.append(self._ingest_dict(input_list[i], schema_list[0], item_path))
                        elif item_type == 'list':
                            valid_list.append(self._ingest_list(input_list[i], schema_list[0], item_path))
                    except:
                        pass
                if isinstance(max_size, int):
                    if len(valid_list) == max_size:
                        return valid_list

        return valid_list