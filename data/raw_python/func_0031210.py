def entry(self,
              name=None,
              dataset=None,
              recommended_full_name=None,
              recommended_short_name=None,
              gene_name=None,
              taxid=None,
              accession=None,
              organism_host=None,
              feature_type=None,
              function_=None,
              ec_number=None,
              db_reference=None,
              alternative_name=None,
              disease_comment=None,
              disease_name=None,
              tissue_specificity=None,
              pmid=None,
              keyword=None,
              subcellular_location=None,
              tissue_in_reference=None,
              sequence=None,
              limit=None,
              as_df=False):
        """Method to query :class:`.models.Entry` objects in database

        An entry is the root element in UniProt datasets. Everything is linked to entry and can be accessed from

        :param name: UniProt entry name(s)
        :type name: str or tuple(str) or None

        :param dataset: Swiss-Prot or TrEMBL
        :type name: str or tuple(str) or None

        :param recommended_full_name: recommended full protein name(s)
        :type recommended_full_name: str or tuple(str) or None

        :param recommended_short_name: recommended short protein name(s)
        :type recommended_short_name: str or tuple(str) or None

        :param tissue_in_reference: tissue(s) mentioned in reference
        :type tissue_in_reference: str or tuple(str) or None

        :param subcellular_location: subcellular location(s)
        :type subcellular_location: str or tuple(str) or None

        :param keyword: keyword(s)
        :type keyword: str or tuple(str) or None

        :param pmid: PubMed identifier(s)
        :type pmid: int or tuple(int) or None

        :param tissue_specificity: tissue specificit(y/ies)
        :type tissue_specificity: str or tuple(str) or None

        :param disease_comment: disease_comment(s)
        :type disease_comment: str or tuple(str) or None

        :param alternative_name: alternative name(s)
        :type alternative_name: str or tuple(str) or None

        :param db_reference: cross reference identifier(s)
        :type db_reference: str or tuple(str) or None

        :param ec_number: enzyme classification number(s), e.g. 1.1.1.1
        :type ec_number: str or tuple(str) or None

        :param function_: description of protein function(s)
        :type function_: str or tuple(str) or None

        :param feature_type: feature type(s)
        :type feature_type: str or tuple(str) or None

        :param organism_host: organism host(s) as taxid(s)
        :type organism_host: int or tuple(int) or None

        :param accession: UniProt accession number(s)
        :type accession: str or tuple(str) or None

        :param disease_name: disease name(s)
        :type disease_name: str or tuple(str) or None

        :param gene_name: gene name(s)
        :type gene_name: str or tuple(str) or None

        :param taxid: NCBI taxonomy identifier(s)
        :type taxid: int or tuple(int) or None

        :param sequence: Amino acid sequence(s)
        :type sequence: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.Entry`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.Entry`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.Entry)

        model_queries_config = (
            (dataset, models.Entry.dataset),
            (name, models.Entry.name),
            (recommended_full_name, models.Entry.recommended_full_name),
            (recommended_short_name, models.Entry.recommended_short_name),
            (gene_name, models.Entry.gene_name),
            (taxid, models.Entry.taxid),
        )
        q = self.get_model_queries(q, model_queries_config)

        one_to_many_queries_config = (
            (accession, models.Accession.accession),
            (organism_host, models.OrganismHost.taxid),
            (feature_type, models.Feature.type_),
            (function_, models.Function.text),
            (ec_number, models.ECNumber.ec_number),
            (db_reference, models.DbReference.identifier),
            (alternative_name, models.AlternativeFullName.name),
            (disease_comment, models.DiseaseComment.comment),
            (tissue_specificity, models.TissueSpecificity.comment),
            (sequence, models.Sequence.sequence),
        )
        q = self.get_one_to_many_queries(q, one_to_many_queries_config)

        many_to_many_queries_config = (
            (pmid, models.Entry.pmids, models.Pmid.pmid),
            (keyword, models.Entry.keywords, models.Keyword.name),
            (subcellular_location, models.Entry.subcellular_locations, models.SubcellularLocation.location),
            (tissue_in_reference, models.Entry.tissue_in_references, models.TissueInReference.tissue)
        )
        q = self.get_many_to_many_queries(q, many_to_many_queries_config)

        if disease_name:
            q = q.join(models.Entry.disease_comments).join(models.DiseaseComment.disease)
            if isinstance(disease_name, str):
                q = q.filter(models.Disease.name.like(disease_name))
            elif isinstance(disease_name, Iterable):
                q = q.filter(models.Disease.name.in_(disease_name))

        return self._limit_and_df(q, limit, as_df)