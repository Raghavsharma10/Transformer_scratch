def _post_resource(self, body):
        """
        Create new resources and associated attributes.

        Example:

            acs.post_resource([
                {
                    "resourceIdentifier": "masaya",
                    "parents": [],
                    "attributes": [
                        {
                            "issuer": "default",
                            "name": "country",
                            "value": "Nicaragua"
                            }
                        ],
                }
            ])

        The issuer is effectively a namespace, and in policy evaluations you
        identify an attribute by a specific namespace.  Many examples provide
        a URL but it could be any arbitrary string.

        The body is a list, so many resources can be added at the same time.
        """
        assert isinstance(body, (list)), "POST for requires body to be a list"
        uri = self._get_resource_uri()
        return self.service._post(uri, body)