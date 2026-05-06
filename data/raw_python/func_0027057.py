def get_fields(self, serializer_fields):
        """
        Get fields metadata skipping empty fields
        """
        fields = OrderedDict()
        for field_name, field in serializer_fields.items():
            # Skip tags field in action because it is needed only for resource creation
            # See also: WAL-1223
            if field_name == 'tags':
                continue
            info = self.get_field_info(field, field_name)
            if info:
                fields[field_name] = info
        return fields