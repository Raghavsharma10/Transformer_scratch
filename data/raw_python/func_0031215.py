def alternative_short_name(self, name=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.AlternativeShortlName` objects in database

        :param name: alternative short name(s)
        :type name: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.AlternativeShortName`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.AlternativeShortName`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.AlternativeShortName)

        model_queries_config = (
            (name, models.AlternativeShortName.name),
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)