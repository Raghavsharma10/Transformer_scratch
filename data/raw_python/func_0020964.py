def update_record(self, orcid_id, token, request_type, data, put_code,
                      content_type='application/orcid+json'):
        """Add a record to a profile.

        Parameters
        ----------
        :param orcid_id: string
            Id of the author.
        :param token: string
            Token received from OAuth 2 3-legged authorization.
        :param request_type: string
            One of 'activities', 'education', 'employment', 'funding',
            'peer-review', 'work'.
        :param data: dict | lxml.etree._Element
            The record in Python-friendly format, as either JSON-compatible
            dictionary (content_type == 'application/orcid+json') or
            XML (content_type == 'application/orcid+xml')
        :param put_code: string
            The id of the record. Can be retrieved using read_record_* method.
            In the result of it, it will be called 'put-code'.
        :param content_type: string
            MIME type of the data being sent.
        """
        self._update_activities(orcid_id, token, requests.put, request_type,
                                data, put_code, content_type)