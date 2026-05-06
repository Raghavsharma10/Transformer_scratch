def build_schema(self, fields, language):
        """
        Build the index schema for the given field. New argument language.
        :param fields:
        :param language: the language code
        :return: a dictionary wit the field name (string) and the
                 mapping configuration (dictionary)
        """
        content_field_name = ''
        mapping = {
            DJANGO_CT: {'type': 'string', 'index': 'not_analyzed', 'include_in_all': False},
            DJANGO_ID: {'type': 'string', 'index': 'not_analyzed', 'include_in_all': False},
        }

        for field_name, field_class in fields.items():
            field_mapping = FIELD_MAPPINGS.get(
                field_class.field_type, DEFAULT_FIELD_MAPPING).copy()
            if field_class.boost != 1.0:
                field_mapping['boost'] = field_class.boost

            if field_class.document is True:
                content_field_name = field_class.index_fieldname

            # Do this last to override `text` fields.
            if field_mapping['type'] == 'string':
                # Use the language analyzer for text fields.
                if field_mapping['analyzer'] == DEFAULT_FIELD_MAPPING['analyzer']:
                    field_mapping['analyzer'] = get_analyzer_for(language)
                if field_class.indexed is False or hasattr(field_class, 'facet_for'):
                    field_mapping['index'] = 'not_analyzed'
                    del field_mapping['analyzer']

            mapping[field_class.index_fieldname] = field_mapping

        return content_field_name, mapping