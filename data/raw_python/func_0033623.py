def parse_json(self, req, name, field):
        """Pull a json value from the request."""
        if not (req.body and is_json_request(req)):
            return core.missing
        json_data = req.json
        if json_data is None:
            return core.missing
        return core.get_value(json_data, name, field, allow_many_nested=True)