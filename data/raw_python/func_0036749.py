def shelter_get(self, **kwargs):
        """
        shelter.get wrapper. Given a shelter ID, retrieve its details in
        dict form.

        :rtype: dict
        :returns: The shelter's details.
        """

        root = self._do_api_call("shelter.get", kwargs)

        shelter = root.find("shelter")
        for field in shelter:
            record = {}
            for field in shelter:
                record[field.tag] = field.text
            return record