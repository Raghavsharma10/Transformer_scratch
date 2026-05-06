def get_cognitive_process_id(self):
        """Gets the grade ``Id`` associated with the cognitive process.

        return: (osid.id.Id) - the grade ``Id``
        raise:  IllegalState - ``has_cognitive_process()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['cognitiveProcessId']):
            raise errors.IllegalState('this Objective has no cognitive_process')
        else:
            return Id(self._my_map['cognitiveProcessId'])