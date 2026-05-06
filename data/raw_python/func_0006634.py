def load_from_json(data):
        """
        Load a :class:`Item` from a dictionary ot string (that will be parsed
        as json)
        """
        if isinstance(data, str):
            data = json.loads(data)
        return Item(data['title'], data['uri'])