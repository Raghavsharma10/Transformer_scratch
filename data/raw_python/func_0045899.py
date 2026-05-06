def get_composition_id(self):
        """Gets the ``Composition``  ``Id`` corresponding to this asset.

        return: (osid.id.Id) - the composiiton ``Id``
        raise:  IllegalState - ``is_composition()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['compositionId']):
            raise errors.IllegalState('composition empty')
        return Id(self._my_map['compositionId'])