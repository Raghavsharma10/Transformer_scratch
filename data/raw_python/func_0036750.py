def shelter_getpets(self, **kwargs):
        """
        shelter.getPets wrapper. Given a shelter ID, retrieve either a list of
        pet IDs (if ``output`` is ``'id'``), or a generator of pet record
        dicts (if ``output`` is ``'full'`` or ``'basic'``).

        :rtype: generator
        :returns: Either a generator of pet ID strings or pet record dicts,
            depending on the value of the ``output`` keyword.
        :raises: :py:exc:`petfinder.exceptions.LimitExceeded` once you have
            reached the maximum number of records your credentials allow you
            to receive.
        """

        def shelter_getpets_parser_ids(root, has_records):
            """
            Parser for output=id.
            """
            pet_ids = root.findall("petIds/id")
            for pet_id in pet_ids:
                yield pet_id.text

        def shelter_getpets_parser_records(root, has_records):
            """
            Parser for output=full or output=basic.
            """
            for pet in root.findall("pets/pet"):
                yield self._parse_pet_record(pet)


        # Depending on the output value, select the correct parser.
        if kwargs.get("output", "id") == "id":
            shelter_getpets_parser = shelter_getpets_parser_ids
        else:
            shelter_getpets_parser = shelter_getpets_parser_records

        return self._do_autopaginating_api_call(
            "shelter.getPets", kwargs, shelter_getpets_parser
        )