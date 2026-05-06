def _join_pathway(query, pathway_id, pathway_name):
        """helper function to add a query join to Pathway model
        
        :param `sqlalchemy.orm.query.Query` query: SQL Alchemy query  
        :param str pathway_id: pathway identifier
        :param str pathway_name: pathway name
        :return: `sqlalchemy.orm.query.Query` object
        """
        if pathway_id or pathway_name:
            if pathway_id:
                query = query.filter(models.Pathway.pathway_id.like(pathway_id))
            if pathway_name:
                query = query.filter(models.Pathway.pathway_name.like(pathway_name))

        return query