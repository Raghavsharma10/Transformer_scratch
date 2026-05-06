def parent(self, parent_object, limit_parent_language=True):
        """
        Return all content items which are associated with a given parent object.
        """
        lookup = get_parent_lookup_kwargs(parent_object)

        # Filter the items by default, giving the expected "objects for this parent" items
        # when the parent already holds the language state.
        if limit_parent_language:
            language_code = get_parent_language_code(parent_object)
            if language_code:
                lookup['language_code'] = language_code

        return self.filter(**lookup)