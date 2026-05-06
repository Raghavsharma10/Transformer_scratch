def pmid(self,
             pmid=None,
             entry_name=None,
             first=None,
             last=None,
             volume=None,
             name=None,
             date=None,
             title=None,
             limit=None,
             as_df=False
             ):
        """Method to query :class:`.models.Pmid` objects in database

        :param pmid: PubMed identifier(s)
        :type pmid: int or tuple(int) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param first: first page(s)
        :type first: str or tuple(str) or None

        :param last: last page(s)
        :type last: str or tuple(str) or None

        :param volume: volume(s)
        :type volume: int or tuple(int) or None

        :param name: name(s) of journal
        :type name: str or tuple(str) or None

        :param date: publication year(s)
        :type date: int or tuple(int) or None

        :param title: title(s) of publication
        :type title: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.Pmid`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.Pmid`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.Pmid)

        model_queries_config = (
            (pmid, models.Pmid.pmid),
            (last, models.Pmid.last),
            (first, models.Pmid.first),
            (volume, models.Pmid.volume),
            (name, models.Pmid.name),
            (date, models.Pmid.date),
            (title, models.Pmid.title)
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_many_to_many_queries(q, ((entry_name, models.Pmid.entries, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)