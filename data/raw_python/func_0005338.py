def patch(self, payload, append_to_arrays=True):
        """
        Patches current record and udpates the current instance's 'attrs'
        attribute to reflect the new changes.

        Args:
            payload - hash. This will be JSON-formatted prior to sending the request.

        Returns:
            `dict`. The JSON formatted response.

        Raises:
            `requests.exceptions.HTTPError`: The status code is not ok.
        """
        if not isinstance(payload, dict):
            raise ValueError("The 'payload' parameter must be provided a dictionary object.")
        payload = self.__class__.set_id_in_fkeys(payload)
        if append_to_arrays:
            for key in payload:
                val = payload[key]
                if type(val) == list:
                    val.extend(getattr(self, key))
                    payload[key] = list(set(val))
        payload = self.check_boolean_fields(payload)
        payload = self.__class__.add_model_name_to_payload(payload)
        self.debug_logger.debug("PATCHING payload {}".format(json.dumps(payload, indent=4)))
        res = requests.patch(url=self.record_url, json=payload, headers=HEADERS, verify=False)
        self.write_response_html_to_file(res,"bob.html")
        res.raise_for_status()
        json_res = res.json()
        self.debug_logger.debug("Success")
        self.attrs = json_res
        return json_res