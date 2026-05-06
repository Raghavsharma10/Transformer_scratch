def not_found(entity_id=None, message='Entity not found'):
        """
        Build a response to indicate that the requested entity was not found.

        :param string message:
            An optional message, defaults to 'Entity not found'
        :param string entity_id:
            An option ID of the entity requested and which was not found
        :return:
            A flask Response object, can be used as a return type from service methods
        """
        resp = jsonify({'message': message, 'entity_id': entity_id})
        resp.status_code = 404
        return resp