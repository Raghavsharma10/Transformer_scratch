def feature(self, type_=None, identifier=None, description=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.Feature` objects in database

        Check available features types with ``pyuniprot.query().feature_types``

        :param type_: type(s) of feature
        :type type_: str or tuple(str) or None

        :param identifier: feature identifier(s)
        :type identifier: str or tuple(str) or None

        :param description: description(s) of feature(s)
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
            - if `as_df == False` -> list(:class:`.models.Feature`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.Feature`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.Feature)

        model_queries_config = (
            (type_, models.Feature.type_),
            (identifier, models.Feature.identifier),
            (description, models.Feature.description)
        )
        q = self.get_model_queries(q, model_queries_config)

        q = self.get_one_to_many_queries(q, ((entry_name, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)