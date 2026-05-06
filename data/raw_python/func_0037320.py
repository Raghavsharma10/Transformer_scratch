def _join_disease(query, disease_definition, disease_id, disease_name):
        """helper function to add a query join to Disease model
        
        :param sqlalchemy.orm.query.Query query: SQL Alchemy query
        :param disease_definition: 
        :param str disease_id: see :attr:`models.Disease.disease_id`
        :param disease_name: 
        :rtype: sqlalchemy.orm.query.Query
        """
        if disease_definition or disease_id or disease_name:
            query = query.join(models.Disease)

            if disease_definition:
                query = query.filter(models.Disease.definition.like(disease_definition))

            if disease_id:
                query = query.filter(models.Disease.disease_id == disease_id)

            if disease_name:
                query = query.filter(models.Disease.disease_name.like(disease_name))

        return query