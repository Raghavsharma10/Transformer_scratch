def clean_value(self, val):
        """
        val =
        :param dict val: {"content":"", "name":"", "ext":"", "type":""}
        :return:
        """
        if isinstance(val, dict):
            if self.random_name:
                val['random_name'] = self.random_name
            if 'file_name' in val.keys():
                val['name'] = val.pop('file_name')
                val['content'] = val.pop('file_content')
            return self.file_manager().store_file(**val)

        # If val is not instance of dict, it should be return itself because the val is the key of
        # the file
        try:
            return str(val)
        except ValueError:
            raise ValidationError("%r could not be cast to string" % val)