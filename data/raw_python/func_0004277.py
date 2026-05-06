def get_rule_by_id(self, rule_id):
        """Get rule by indentifier.

        :param rule_id: Rule identifier

        :return: Dictionary with the following structure:

        ::

            {'rule': {'environment': < environment_id >,
            'content': < content >,
            'custom': < custom >,
            'id': < id >,
            'name': < name >}}

        :raise UserNotAuthorizedError: User dont have permition.
        :raise InvalidParameterError: RULE identifier is null or invalid.
        :raise DataBaseError: Can't connect to networkapi database.
        """

        url = "rule/get_by_id/" + str(rule_id)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, ['rule_contents', 'rule_blocks'])