def set_text(self, text=None):
        """stub"""
        if text is None:
            raise NullArgument()
        if self.get_text_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_string(
                text,
                self.get_text_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['text']['text'] = text