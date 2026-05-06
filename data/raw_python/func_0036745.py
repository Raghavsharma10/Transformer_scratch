def pet_get(self, **kwargs):
        """
        pet.get wrapper. Returns a record dict for the requested pet.

        :rtype: dict
        :returns: The pet's record dict.
        """
        root = self._do_api_call("pet.get", kwargs)

        return self._parse_pet_record(root.find("pet"))