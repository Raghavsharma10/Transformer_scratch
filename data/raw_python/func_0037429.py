def create(self, subscription_id, name, parameters, type='analysis', service='facebook'):
        """ Create a PYLON task

            :param subscription_id: The ID of the recording to create the task for
            :type subscription_id: str
            :param name: The name of the new task
            :type name: str
            :param parameters: The parameters for this task
            :type parameters: dict
            :param type: The type of analysis to create, currently only 'analysis' is accepted
            :type type: str
            :param service: The PYLON service (facebook)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {
            'subscription_id': subscription_id,
            'name': name,
            'parameters': parameters,
            'type': type
        }

        return self.request.post(service + '/task/', params)