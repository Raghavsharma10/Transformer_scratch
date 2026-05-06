def match_resource_id(self, resource_id, match):
        """Sets the resource ``Id`` for this query.

        arg:    resource_id (osid.id.Id): a resource ``Id``
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  NullArgument - ``resource_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(resource_id, Id):
            raise errors.InvalidArgument()
        self._add_match('resourceId', str(resource_id), match)