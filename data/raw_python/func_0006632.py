def load_from_json(data):
        """
        Load a :class:`RegistryReponse` from a dictionary or a string (that
        will be parsed as json).
        """
        if isinstance(data, str):
            data = json.loads(data)
        applications = [
            ApplicationResponse.load_from_json(a) for a in data['applications']
        ] if data['applications'] is not None else []
        return RegistryResponse(
            data['query_uri'], data['success'],
            data['has_references'], data['count'], applications
        )