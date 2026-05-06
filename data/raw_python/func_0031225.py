def tissue_in_reference(self, tissue=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.TissueInReference` objects in database

        :param tissue: tissue(s) linked to reference
        :type tissue: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.TissueInReference`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.TissueInReference`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.TissueInReference)

        model_queries_config = (
            (tissue, models.TissueInReference.tissue),
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_many_to_many_queries(q, ((entry_name, models.TissueInReference.entries, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)