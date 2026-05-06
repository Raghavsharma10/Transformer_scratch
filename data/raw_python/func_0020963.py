def remove_record(self, orcid_id, token, request_type, put_code):
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
         :param put_code: string
            The id of the record. Can be retrieved using read_record_* method.
            In the result of it, it will be called 'put-code'.
        """
        self._update_activities(orcid_id, token, requests.delete, request_type,
                                put_code=put_code)