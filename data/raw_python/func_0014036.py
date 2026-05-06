def jsonify_payload(self):
        """ Dump the payload to JSON """
        # Assume already json serialized
        if isinstance(self.payload, string_types):
            return self.payload
        return json.dumps(self.payload, cls=StandardJSONEncoder)