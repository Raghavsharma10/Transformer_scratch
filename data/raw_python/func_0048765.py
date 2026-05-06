def match_ancestor_bank_id(self, bank_id, match):
        """Sets the bank ``Id`` for to match banks in which the specified bank is an acestor.

        arg:    bank_id (osid.id.Id): a bank ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``bank_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # matches when the bank_id param is an ancestor of
        # any bank
        bank_descendants = self._get_descendant_catalog_ids(bank_id)
        identifiers = [ObjectId(i.identifier) for i in bank_descendants]
        self._query_terms['_id'] = {'$in': identifiers}