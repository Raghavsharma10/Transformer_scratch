def _has_nested(self, relations, operator='>=', count=1, boolean='and', extra=None):
        """
        Add nested relationship count conditions to the query.

        :param relations: nested relations
        :type relations: str

        :param operator: The operator
        :type operator: str

        :param count: The count
        :type count: int

        :param boolean: The boolean value
        :type boolean: str

        :param extra: The extra query
        :type extra: Builder or callable

        :rtype: Builder
        """
        relations = relations.split('.')

        def closure(q):
            if len(relations) > 1:
                q.where_has(relations.pop(0), closure)
            else:
                q.has(relations.pop(0), operator, count, boolean, extra)

        return self.where_has(relations.pop(0), closure)