def tissue_specificity(self, comment=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.TissueSpecificity` objects in database

        Provides information on the expression of a gene at the mRNA or protein level in cells or in tissues of
        multicellular organisms. By default, the information is derived from experiments at the mRNA level, unless
        specified ‘at protein level

        :param comment: Comment(s) describing tissue specificity
        :type comment: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.TissueSpecificity`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.TissueSpecificity`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.TissueSpecificity)

        q = self.get_model_queries(q, ((comment, models.TissueSpecificity.comment),))

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)