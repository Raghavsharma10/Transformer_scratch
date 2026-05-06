def get(cls, object_id):
        """
        Retrieves a single model

        :param object_id: the primary id of the model
        :type object_id: integer
        :return: the object of the parsed xml object
        :rtype: object
        """
        return fields.ObjectField(name=cls.ENDPOINT, init_class=cls).decode(
            cls.element_from_string(
                cls._get_request(endpoint=cls.ENDPOINT + '/' + str(object_id)).text
            )
        )