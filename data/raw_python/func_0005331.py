def _get(self, rec_id=None, upstream=None):
        """
        Fetches a record by the record's ID or upstream_identifier.

        Raises:
            `pulsarpy.models.RecordNotFound`: A record could not be found.
        """
        if rec_id:
            self.record_url = self.__class__.get_record_url(rec_id)
            self.debug_logger.debug("GET {} record with ID {}: {}".format(self.__class__.__name__, rec_id, self.record_url))
            response = requests.get(url=self.record_url, headers=HEADERS, verify=False)
            if not response.ok and response.status_code == requests.codes.NOT_FOUND:
                raise RecordNotFound("Search for {} record with ID '{}' returned no results.".format(self.__class__.__name__, rec_id))
            self.write_response_html_to_file(response,"get_bob.html")
            response.raise_for_status()
            return response.json()
        elif upstream:
            rec_json = self.__class__.find_by({"upstream_identifier": upstream}, require=True)
            self.record_url = self.__class__.get_record_url(rec_json["id"])
        return rec_json