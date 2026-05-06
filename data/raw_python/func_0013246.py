def featuresQuery(self, **kwargs):
        """
        Converts a dictionary of keyword arguments into a tuple
        of SQL select statements and the list of SQL arguments
        """
        # TODO: Optimize by refactoring out string concatenation
        sql = ""
        sql_rows = "SELECT * FROM FEATURE WHERE id > 1 "
        sql_args = ()
        if 'name' in kwargs and kwargs['name']:
            sql += "AND name = ? "
            sql_args += (kwargs.get('name'),)
        if 'geneSymbol' in kwargs and kwargs['geneSymbol']:
            sql += "AND gene_name = ? "
            sql_args += (kwargs.get('geneSymbol'),)
        if 'start' in kwargs and kwargs['start'] is not None:
            sql += "AND end > ? "
            sql_args += (kwargs.get('start'),)
        if 'end' in kwargs and kwargs['end'] is not None:
            sql += "AND start < ? "
            sql_args += (kwargs.get('end'),)
        if 'referenceName' in kwargs and kwargs['referenceName']:
            sql += "AND reference_name = ?"
            sql_args += (kwargs.get('referenceName'),)
        if 'parentId' in kwargs and kwargs['parentId']:
            sql += "AND parent_id = ? "
            sql_args += (kwargs['parentId'],)
        if kwargs.get('featureTypes') is not None \
                and len(kwargs['featureTypes']) > 0:
            sql += "AND type IN ("
            sql += ", ".join(["?", ] * len(kwargs.get('featureTypes')))
            sql += ") "
            sql_args += tuple(kwargs.get('featureTypes'))
        sql_rows += sql
        sql_rows += " ORDER BY reference_name, start, end ASC "
        return sql_rows, sql_args