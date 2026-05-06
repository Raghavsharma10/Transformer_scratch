def _get_id(self, id_, pkg_name):
        """
        Returns the primary id given an alias.

        If the id provided is not in the alias table, it will simply be
        returned as is.

        Only looks within the Id Alias namespace for the session package

        """
        collection = JSONClientValidated('id',
                                         collection=pkg_name + 'Ids',
                                         runtime=self._runtime)
        try:
            result = collection.find_one({'aliasIds': {'$in': [str(id_)]}})
        except errors.NotFound:
            return id_
        else:
            return Id(result['_id'])