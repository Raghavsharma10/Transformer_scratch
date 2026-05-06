def _ingest_number(self, input_number, path_to_root):

        '''
            a helper method for ingesting a number

        :return: valid_number
        '''

        valid_number = 0.0

        try:
            valid_number = self._validate_number(input_number, path_to_root)
        except:
            rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
            if 'default_value' in self.keyMap[rules_path_to_root]:
                valid_number = self.keyMap[rules_path_to_root]['default_value']
            elif 'integer_data' in self.keyMap[rules_path_to_root].keys():
                if self.keyMap[rules_path_to_root]['integer_data']:
                    valid_number = 0

        return valid_number