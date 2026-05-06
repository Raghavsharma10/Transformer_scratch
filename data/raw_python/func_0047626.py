def set_solution(self, text):
        """stub"""
        if not self.my_osid_object_form._is_valid_string(
                text, self.get_solution_metadata()):
            raise InvalidArgument('text')
        if is_string(text):
            self.my_osid_object_form._my_map['solution'] = {
                'text': text,
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE)
            }
        else:
            self.my_osid_object_form._my_map['solution'] = text