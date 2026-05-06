def get(cls, object_id):
        """
        Get all parties (people and companies) associated with a given tag.
        :param object_id: the primary id of the model
        :type object_id: integer
        :return: the parties
        :rtype: list
        """
        from highton.models.party import Party
        return fields.ListField(name=Party.ENDPOINT, init_class=Party).decode(
            cls.element_from_string(
                cls._get_request(endpoint=cls.ENDPOINT + '/' + str(object_id)).text
            )
        )