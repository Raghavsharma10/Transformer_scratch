def find_by_or(cls, payload):
        """
        Searches the model in question by OR joining the query parameters.

        Implements a Railsy way of looking for a record using a method by the same name and passing
        in the query as a string (for the OR operator joining to be specified).

        Only the first hit is returned, and there is not particular ordering specified in the server-side
        API method.

        Args:
            payload: `dict`. The attributes of a record to search for by using OR operator joining
                for each query parameter.

        Returns:
            `dict`: The JSON serialization of the record, if any, found by the API call.
            `None`: If the API call didnt' return any results.
        """
        if not isinstance(payload, dict):
            raise ValueError("The 'payload' parameter must be provided a dictionary object.")
        url = os.path.join(cls.URL, "find_by_or")
        payload = {"find_by_or": payload}
        cls.debug_logger.debug("Searching Pulsar {} for {}".format(cls.__name__, json.dumps(payload, indent=4)))
        res = requests.post(url=url, json=payload, headers=HEADERS, verify=False)
        cls.write_response_html_to_file(res,"bob.html")
        if res:
           try:
               res = res[cls.MODEL_NAME]
           except KeyError:
               # Key won't be present if there isn't a serializer for it on the server.
               pass
        return res