def get_knowledge_category_id(self):
        """Gets the grade ``Id`` associated with the knowledge dimension.

        return: (osid.id.Id) - the grade ``Id``
        raise:  IllegalState - ``has_knowledge_category()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['knowledgeCategoryId']):
            raise errors.IllegalState('this Objective has no knowledge_category')
        else:
            return Id(self._my_map['knowledgeCategoryId'])