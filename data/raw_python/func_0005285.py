def _get_relation(self, related_model: type, relations: List[str]) -> Tuple[Optional[List[type]], Optional[type]]:
        """Transform the list of relation to list of class.

        :param related_mode: The model of the query.
        :type related_mode: type

        :param relations: The relation list get from the `_extract_relations`.
        :type relations: List[str]

        :return: Tuple with the list of relations (class) and the second
            element is the last relation class.
        :rtype: Tuple[Optional[List[type]], Optional[type]]
        """
        relations_list, last_relation = [], related_model
        for relation in relations:
            relationship = getattr(last_relation, relation, None)
            if relationship is None:
                return (None, None)
            last_relation = relationship.mapper.class_
            relations_list.append(last_relation)
        return (relations_list, last_relation)