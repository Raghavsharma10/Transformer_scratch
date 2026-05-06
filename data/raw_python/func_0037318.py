def _join_gene(query, gene_name, gene_symbol, gene_id):
        """helper function to add a query join to Gene model

        :param `sqlalchemy.orm.query.Query` query: SQL Alchemy query 
        :param str gene_name: gene name
        :param str gene_symbol: gene symbol
        :param int gene_id: NCBI Gene identifier
        :return: `sqlalchemy.orm.query.Query` object
        """
        if gene_name or gene_symbol:
            query = query.join(models.Gene)

            if gene_symbol:
                query = query.filter(models.Gene.gene_symbol.like(gene_symbol))

            if gene_name:
                query = query.filter(models.Gene.gene_name.like(gene_name))

            if gene_id:
                query = query.filter(models.Gene.gene_id.like(gene_id))

        return query