def get_where_clause_from_dict(dictionary, join_operator='AND'):
        """Builds a where clause from a dictionary
        """
        CoyoteDb.escape_dictionary(dictionary)
        clause = join_operator.join(
            (' {k} is {v} ' if str(v).lower() == 'null' else ' {k} = {v} ').format(k=k, v=v)  # IS should be the operator for null values
            for k, v in dictionary.iteritems())
        return clause