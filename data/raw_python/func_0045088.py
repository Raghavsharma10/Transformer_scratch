def _format_listnodes(self, listnodes):
        """
        Format ListNodes and their fields into tuples that can be printed with _print_fields().
        """
        fields = list()
        for name, node in listnodes:
            fields.append(('--', '', '', '--'))
            fields.append(('', '**%s(ListNode)**' % name, '', ''))
            for link in node.get_links():
                linked_model = link['mdl'](super_context)
                null = self._marker_true if link['null'] is True else self._marker_false
                fields.append((self._nodelist_field_prefix, link['field'],
                               '%s()' % linked_model.title, null))
            fields.extend(self._get_model_fields(node, self._nodelist_field_prefix))
        return fields