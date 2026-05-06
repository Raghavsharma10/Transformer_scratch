def _ingest_boolean(self, input_boolean, path_to_root):

        '''
            a helper method for ingesting a boolean

        :return: valid_boolean
        '''

        valid_boolean = False

        try:
            valid_boolean = self._validate_boolean(input_boolean, path_to_root)
        except:
            rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
            if 'default_value' in self.keyMap[rules_path_to_root]:
                valid_boolean = self.keyMap[rules_path_to_root]['default_value']

        return valid_boolean