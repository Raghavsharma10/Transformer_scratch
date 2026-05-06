def _format_links_fields(self, links):
        """
        Format the fields containing links into 4-tuples printable by _print_fields().
        """
        fields = list()
        for link in links:
            linked_model = link['mdl'](super_context)
            null = self._marker_true if link['null'] is True else self._marker_false
            # In LinkProxy, if reverse_name is empty then only reverse has the name
            # of the field on the link_source side
            field_name = link['field'] or link['reverse']
            fields.append((self._field_prefix, field_name, '%s()' % linked_model.title, null))
        fields.sort(key=lambda f: f[1])
        return fields