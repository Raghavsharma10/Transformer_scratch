def _ingest_string(self, input_string, path_to_root):

        '''
            a helper method for ingesting a string

        :return: valid_string
        '''

        valid_string = ''

        try:
            valid_string = self._validate_string(input_string, path_to_root)
        except:
            rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
            if 'default_value' in self.keyMap[rules_path_to_root]:
                valid_string = self.keyMap[rules_path_to_root]['default_value']

        return valid_string