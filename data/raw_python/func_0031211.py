def disease(self,
                identifier=None,
                ref_id=None,
                ref_type=None,
                name=None,
                acronym=None,
                description=None,
                entry_name=None,
                limit=None,
                as_df=False
                ):
        """Method to query :class:`.models.Disease` objects in database


        :param identifier: disease UniProt identifier(s)
        :type identifier: str or tuple(str) or None

        :param ref_id: identifier(s) of referenced database
        :type ref_id: str or tuple(str) or None

        :param ref_type: database name(s)
        :type ref_type: str or tuple(str) or None

        :param name: disease name(s)
        :type name: str or tuple(str) or None

        :param acronym: disease acronym(s)
        :type acronym: str or tuple(str) or None

        :param description: disease description(s)
        :type description: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.Disease`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.Disease`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.Disease)

        model_queries_config = (
            (identifier, models.Disease.identifier),
            (ref_id, models.Disease.ref_id),
            (ref_type, models.Disease.ref_type),
            (name, models.Disease.name),
            (acronym, models.Disease.acronym),
            (description, models.Disease.description)
        )
        q = self.get_model_queries(q, model_queries_config)

        if entry_name:
            q = q.session.query(models.Disease).join(models.DiseaseComment).join(models.Entry)
            if isinstance(entry_name, str):
                q = q.filter(models.Entry.name == entry_name)
            elif isinstance(entry_name, Iterable):
                q = q.filter(models.Entry.name.in_(entry_name))

        return self._limit_and_df(q, limit, as_df)