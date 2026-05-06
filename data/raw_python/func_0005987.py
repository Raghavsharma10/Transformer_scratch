def get_update_clause_from_dict(dictionary, datetime_format='%Y-%m-%d %H:%M:%S'):
        """Builds the update values clause of an update statement based on the dictionary representation of an
        instance"""
        items = []

        CoyoteDb.escape_dictionary(dictionary, datetime_format=datetime_format)
        for k,v in dictionary.iteritems():
            item = '{k} = {v}'.format(k=k, v=v)
            items.append(item)
        clause = ', '.join(item for item in items)
        return clause