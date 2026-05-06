def organism_host(self, taxid=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.OrganismHost` objects in database

        :param taxid: NCBI taxonomy identifier(s)
        :type taxid: int or tuple(int) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.OrganismHost`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.OrganismHost`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.OrganismHost)

        q = self.get_model_queries(q, ((taxid, models.OrganismHost.taxid),))

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)