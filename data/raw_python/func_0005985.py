def get_insert_fields_and_values_from_dict(dictionary, datetime_format='%Y-%m-%d %H:%M:%S', db_escape=True):
        """Formats a dictionary to strings of fields and values for insert statements

        @param dictionary: The dictionary whose keys and values are to be inserted
        @param db_escape: If true, will db escape values
        @return: Tuple of strings containing string fields and values, e.g. ('user_id, username', '5, "pandaman"')
        """
        if db_escape:
            CoyoteDb.escape_dictionary(dictionary, datetime_format=datetime_format)

        fields = get_delimited_string_from_list(dictionary.keys(), delimiter=',')  # keys have no quotes
        vals = get_delimited_string_from_list(dictionary.values(), delimiter=',')  # strings get quotes

        return fields, vals