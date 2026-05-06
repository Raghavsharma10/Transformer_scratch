def populated_column_map(self):
        '''Return the _column_map without unused optional fields'''
        column_map = []
        cls = self.model
        for csv_name, field_pattern in cls._column_map:
            # Separate the local field name from foreign columns
            if '__' in field_pattern:
                field_name = field_pattern.split('__', 1)[0]
            else:
                field_name = field_pattern

            # Handle point fields
            point_match = re_point.match(field_name)
            if point_match:
                field = None
            else:
                field = cls._meta.get_field(field_name)

            # Only add optional columns if they are used in the records
            if field and field.blank and not field.has_default():
                kwargs = {field_name: get_blank_value(field)}
                if self.exclude(**kwargs).exists():
                    column_map.append((csv_name, field_pattern))
            else:
                column_map.append((csv_name, field_pattern))
        return column_map