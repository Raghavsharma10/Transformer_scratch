def match_qualifier_id(self, qualifier_id, match):
        """Matches the qualifier identified by the given ``Id``.

        arg:    qualifier_id (osid.id.Id): the Id of the ``Qualifier``
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  NullArgument - ``qualifier_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('qualifierId', str(qualifier_id), bool(match))