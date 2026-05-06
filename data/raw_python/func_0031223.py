def subcellular_location(self, location=None, entry_name=None, limit=None, as_df=False):
        """Method to query :class:`.models.SubcellularLocation` objects in database

        :param location: subcellular location(s)
        :type location: str or tuple(str) or None

        :param entry_name: name(s) in :class:`.models.Entry`
        :type entry_name: str or tuple(str) or None

        :param limit:
            - if `isinstance(limit,int)==True` -> limit
            - if `isinstance(limit,tuple)==True` -> format:= tuple(page_number, results_per_page)
            - if limit == None -> all results
        :type limit: int or tuple(int) or None

        :param bool as_df: if `True` results are returned as :class:`pandas.DataFrame`

        :return:
            - if `as_df == False` -> list(:class:`.models.SubcellularLocation`)
            - if `as_df == True`  -> :class:`pandas.DataFrame`
        :rtype: list(:class:`.models.SubcellularLocation`) or :class:`pandas.DataFrame`
        """
        q = self.session.query(models.SubcellularLocation)

        q = self.get_model_queries(q, ((location, models.SubcellularLocation.location),))

        q = self.get_many_to_many_queries(q, ((entry_name, models.SubcellularLocation.entries, models.Entry.name),))

        return self._limit_and_df(q, limit, as_df)