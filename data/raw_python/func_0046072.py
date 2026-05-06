def add_text(self, text, label=None):
        """stub"""
        if label is None:
            label = self._label_metadata['default_string_values'][0]
        else:
            if not self.my_osid_object_form._is_valid_string(
                    label, self.get_label_metadata()) or '.' in label:
                raise InvalidArgument('label')
        if text is None:
            raise NullArgument('text cannot be none')
        if not (self.my_osid_object_form._is_valid_string(
                text, self.get_text_metadata()) or isinstance(text, DisplayText)):
            raise InvalidArgument('text')
        if utilities.is_string(text):
            self.my_osid_object_form._my_map['texts'][label] = {
                'text': text,
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE)
            }
        else:
            self.my_osid_object_form._my_map['texts'][label] = {
                'text': text.text,
                'languageTypeId': str(text.language_type),
                'scriptTypeId': str(text.script_type),
                'formatTypeId': str(text.format_type)
            }