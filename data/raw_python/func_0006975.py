def _ids_and_column_names(names, force_lower_case=False):
        """Ensure all column names are unique identifiers."""
        fixed = OrderedDict()
        for name in names:
            identifier = RowWrapper._make_identifier(name)
            if force_lower_case:
                identifier = identifier.lower()
            while identifier in fixed:
                identifier = RowWrapper._increment_numeric_suffix(identifier)
            fixed[identifier] = name
        return fixed