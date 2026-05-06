def other_gene_name(self, type_=None, name=None, entry_name=None, limit=None, as_df=None):
        """Method to query :class:`.models.OtherGeneName` objects in database

        :param type_: type(s) of gene name e.g. *synonym*
        :type type_: str or tuple(str) or None

        :param name: other gene name(s)
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
            - if `as_df == False` -> list(:class:`.models.OtherGeneName`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.OtherGeneName`) or :class:`pandas.DataFrame`

        """
        q = self.session.query(models.OtherGeneName)

        model_queries_config = (
            (type_, models.OtherGeneName.type_),
            (name, models.OtherGeneName.name),
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)