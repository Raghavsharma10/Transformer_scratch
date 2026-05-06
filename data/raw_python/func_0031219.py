def db_reference(self, type_=None, identifier=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.DbReference` objects in database

        Check list of available databases with on :py:attr:`.dbreference_types`

        :param type_: type(s) (or name(s)) of database
        :type type_: str or tuple(str) or None

        :param identifier: unique identifier(s) in specific database (type)
        :type identifier: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.DbReference`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.DbReference`) or :class:`pandas.DataFrame`

        **Links**

            - `UniProt dbxref <http://www.uniprot.org/docs/dbxref>`_
        """
        q = self.session.query(models.DbReference)

        model_queries_config = (
            (type_, models.DbReference.type_),
            (identifier, models.DbReference.identifier)
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)