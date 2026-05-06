def find_by(cls, payload, require=False):
        """
        Searches the model in question by AND joining the query parameters.

        Implements a Railsy way of looking for a record using a method by the same name and passing
        in the query as a dict. as well. Only the first hit is returned, and there is no particular
        ordering specified in the server-side API method.

        Args:
            payload: `dict`. The attributes of a record to restrict the search to.
            require: `bool`. True means to raise a `pulsarpy.models.RecordNotFound` exception if no
                record is found.

        Returns:
            `dict`: The JSON serialization of the record, if any, found by the API call.
            `None`: If the API call didnt' return any results. 

        Raises:
            `pulsarpy.models.RecordNotFound`: No records were found, and the `require` parameter is
                True.
        """
        if not isinstance(payload, dict):
            raise ValueError("The 'payload' parameter must be provided a dictionary object.")
        url = os.path.join(cls.URL, "find_by")
        payload = {"find_by": payload}
        cls.debug_logger.debug("Searching Pulsar {} for {}".format(cls.__name__, json.dumps(payload, indent=4)))
        res = requests.post(url=url, json=payload, headers=HEADERS, verify=False)
        #cls.write_response_html_to_file(res,"bob.html")
        res.raise_for_status()
        res_json = res.json()
        if res_json:
           try:
               res_json = res_json[cls.MODEL_NAME]
           except KeyError:
               # Key won't be present if there isn't a serializer for it on the server.
               pass
        else:
            if require:
                raise RecordNotFound("Can't find any {} records with search criteria: '{}'.".format(cls.__name__, payload))
        return res_json