def _join_chemical(query, cas_rn, chemical_id, chemical_name, chemical_definition):
        """helper function to add a query join to Chemical model
        
        :param `sqlalchemy.orm.query.Query` query: SQL Alchemy query 
        :param cas_rn: 
        :param chemical_id: 
        :param chemical_name: 
        :param chemical_definition: 
        :return: `sqlalchemy.orm.query.Query` object 
        """
        if cas_rn or chemical_id or chemical_name or chemical_definition:
            query = query.join(models.Chemical)

            if cas_rn:
                query = query.filter(models.Chemical.cas_rn.like(cas_rn))

            if chemical_id:
                query = query.filter(models.Chemical.chemical_id == chemical_id)

            if chemical_name:
                query = query.filter(models.Chemical.chemical_name.like(chemical_name))

            if chemical_definition:
                query = query.filter(models.Chemical.definition.like(chemical_definition))

        return query