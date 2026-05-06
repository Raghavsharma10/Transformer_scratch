def read_record_public(self, orcid_id, request_type, token, put_code=None,
                           accept_type='application/orcid+json'):
        """Get the public info about the researcher.

        Parameters
        ----------
        :param orcid_id: string
            Id of the queried author.
        :param request_type: string
            For example: 'record'.
            See https://members.orcid.org/api/tutorial/read-orcid-records
            for possible values.
        :param token: string
            Token received from OAuth 2 3-legged authorization.
        :param put_code: string | list of strings
            The id of the queried work. In case of 'works' request_type
            might be a list of strings
        :param accept_type: expected MIME type of received data

        Returns
        -------
        :returns: dict | lxml.etree._Element
            Record(s) in JSON-compatible dictionary representation or
            in XML E-tree, depending on accept_type specified.
        """
        return self._get_info(orcid_id, self._get_public_info, request_type,
                              token, put_code, accept_type)