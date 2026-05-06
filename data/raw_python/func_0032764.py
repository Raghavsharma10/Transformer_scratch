def act_on_droplets(self, **data):
        r"""
        Perform an arbitrary action on all of the droplets to which the tag is
        applied.  ``data`` will be serialized as JSON and POSTed to the proper
        API endpoint.  All currently-documented actions require the POST body
        to be a JSON object containing, at a minimum, a ``"type"`` field.

        :return: a generator of `Action`\ s representing the in-progress
            operations on the droplets
        :rtype: generator of `Action`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return map(api._action, api.request('/v2/droplets/actions', method='POST', params={"tag_name": self.name}, data=data)["actions"])