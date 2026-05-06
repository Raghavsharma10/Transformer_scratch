def load_from_json(data):
        """
        Load a :class:`ApplicationResponse` from a dictionary or string (that
        will be parsed as json).
        """
        if isinstance(data, str):
            data = json.loads(data)
        items = [Item.load_from_json(a) for a in data['items']] if data['items'] is not None else []
        return ApplicationResponse(
            data['title'], data['uri'], data['service_url'],
            data['success'], data['has_references'], data['count'], items
        )