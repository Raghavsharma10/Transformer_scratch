def add_record(self, orcid_id, token, request_type, data,
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
        :param content_type: string
            MIME type of the passed record.

        Returns
        -------
        :returns: string
            Put-code of the new work.
        """
        return self._update_activities(orcid_id, token, requests.post,
                                       request_type, data,
                                       content_type=content_type)