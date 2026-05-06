def pet_getrandom(self, **kwargs):
        """
        pet.getRandom wrapper. Returns a record dict or Petfinder ID
        for a random pet.

        :rtype: dict or str
        :returns: A dict of pet data if ``output`` is ``'basic'`` or ``'full'``,
            and a string if ``output`` is ``'id'``.
        """
        root = self._do_api_call("pet.getRandom", kwargs)

        output_brevity = kwargs.get("output", "id")

        if output_brevity == "id":
            return root.find("petIds/id").text
        else:
            return self._parse_pet_record(root.find("pet"))