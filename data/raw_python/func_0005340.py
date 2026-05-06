def post(cls, payload):
        """Posts the data to the specified record.

        Args:
            payload: `dict`. This will be JSON-formatted prior to sending the request.

        Returns:
            `dict`. The JSON formatted response.

        Raises:
            `Requests.exceptions.HTTPError`: The status code is not ok.
            `RecordNotUnique`: The Rails server returned the exception ActiveRecord::RecordNotUnique.
        """
        if not isinstance(payload, dict):
            raise ValueError("The 'payload' parameter must be provided a dictionary object.")
        payload = cls.set_id_in_fkeys(payload)
        payload = cls.check_boolean_fields(payload)
        payload = cls.add_model_name_to_payload(payload)
        # Run any pre-post hooks:
        payload = cls.prepost_hooks(payload)
        cls.debug_logger.debug("POSTING payload {}".format(json.dumps(payload, indent=4)))
        res = requests.post(url=cls.URL, json=(payload), headers=HEADERS, verify=False)
        cls.write_response_html_to_file(res,"bob.html")
        if not res.ok:
            cls.log_error(res.text)
            res_json = res.json()
            if "exception" in res_json:
                exc_type = res_json["exception"]
                if exc_type == "ActiveRecord::RecordNotUnique":
                    raise RecordNotUnique()
        res.raise_for_status()
        res = res.json()
        cls.log_post(res)
        cls.debug_logger.debug("Success")
        return res