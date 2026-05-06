def match_function_id(self, function_id, match):
        """Matches the function identified by the given ``Id``.

        arg:    function_id (osid.id.Id): the Id of the ``Function``
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  NullArgument - ``function_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('functionId', str(function_id), bool(match))