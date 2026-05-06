def add_markdown(self, markdown):
        """stub"""
        if markdown is None:
            raise NullArgument('markdown cannot be None')
        if not self.my_osid_object_form._is_valid_string(
                markdown, self.get_markdown_metadata()):
            raise InvalidArgument('markdown')
        self.my_osid_object_form._my_map['markdown'] = markdown