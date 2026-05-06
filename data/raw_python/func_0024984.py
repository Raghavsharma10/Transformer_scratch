def _post_subject(self, body):
        """
        Create new subjects and associated attributes.

        Example:

            acs.post_subject([
                {
                    "subjectIdentifier": "/role/evangelist",
                    "parents": [],
                    "attributes": [
                        {
                            "issuer": "default",
                            "name": "role",
                            "value": "developer evangelist",
                        }
                    ]
                }
            ])

        The issuer is effectively a namespace, and in policy evaluations
        you identify an attribute by a specific namespace.  Many examples
        provide a URL but it could be any arbitrary string.

        The body is a list, so many subjects can be added at the same time.
        """
        assert isinstance(body, (list)), "POST requires body to be a list"

        uri = self._get_subject_uri()
        return self.service._post(uri, body)