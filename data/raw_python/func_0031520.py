def from_list(cls, gene_ontology, l):
        """Initialize a `GOAnnotation` object from a list (in GAF2.1 order).
        
        TODO: docstring
        """ 
        assert isinstance(gene_ontology, GeneOntology)
        assert isinstance(l, list)

        assert len(l) == 17

        go_term = gene_ontology[l[4]]

        qualifier = l[3] or []
        with_from = l[7] or None
        db_name = l[9] or None
        db_syn = l[10] or []
        ext = l[15] or []
        product_id = l[16] or None

        annotation = cls(
            db=l[0],
            db_id=l[1],
            db_symbol=l[2],
            go_term=go_term,
            db_ref=l[5],
            ev_code=l[6],
            db_type=l[11],
            taxon=l[12],
            date=l[13],
            assigned_by=l[14],

            qualifier=qualifier,
            with_from=with_from,
            db_name=db_name,
            db_syn=db_syn,
            ext=ext,
            product_id=product_id,
        )
        return annotation