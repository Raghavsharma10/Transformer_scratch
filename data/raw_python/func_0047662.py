def set_source(self, source):
        """stub"""
        if not is_string(source):
            raise InvalidArgument('source value must be a string')
        self.my_osid_object_form._my_map['texts']['source']['text'] = source