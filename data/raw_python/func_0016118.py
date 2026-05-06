def to_json(self, minimal=True):
        """Encode an object as a JSON string.

        :param bool minimal: Construct a minimal representation of the object (ignore nulls and empty collections)

        :rtype: str
        """
        if minimal:
            return json.dumps(self.json_repr(minimal=True), cls=MarathonMinimalJsonEncoder, sort_keys=True)
        else:
            return json.dumps(self.json_repr(), cls=MarathonJsonEncoder, sort_keys=True)