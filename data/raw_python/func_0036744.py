def breed_list(self, **kwargs):
        """
        breed.list wrapper. Returns a list of breed name strings.

        :rtype: list
        :returns: A list of breed names.
        """

        root = self._do_api_call("breed.list", kwargs)

        breeds = []
        for breed in root.find("breeds"):
            breeds.append(breed.text)
        return breeds