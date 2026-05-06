def choices(self, cl):
        """
        Take choices from field's 'choices' attribute for 'ChoicesField' and
        use 'flatchoices' as usual for other fields.
        """
        #: Just tidy up standard implementation for the sake of DRY principle.
        def _choice_item(is_selected, query_string, title):
            return {
                'selected': is_selected,
                'query_string': query_string,
                'display': force_text(title),
            }

        yield _choice_item(
            self.lookup_val is None,
            cl.get_query_string({}, [self.lookup_kwarg]),
            _('All'))

        container = (self.field.choices
                     if isinstance(self.field, ChoicesField) else
                     self.field.flatchoices)

        for lookup, title in container:
            yield _choice_item(
                smart_text(lookup) == self.lookup_val,
                cl.get_query_string({self.lookup_kwarg: lookup}),
                title)