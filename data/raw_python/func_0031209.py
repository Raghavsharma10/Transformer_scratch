def keyword(self, name=None, identifier=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.Keyword` objects in database

        :param name: keyword name(s)
        :type name: str or tuple(str) or None

        :param identifier: keyword identifier(s)
        :type identifier: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type identifier: str or tuple(str) or None


        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.Keyword`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.Keyword`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.Keyword)

        model_queries_config = (
            (name, models.Keyword.name),
            (identifier, models.Keyword.identifier)
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_many_to_many_queries(q, ((entry_name, models.Keyword.entries, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)