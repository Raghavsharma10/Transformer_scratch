def pet_find(self, **kwargs):
        """
        pet.find wrapper. Returns a generator of pet record dicts
        matching your search criteria.

        :rtype: generator
        :returns: A generator of pet record dicts.
        :raises: :py:exc:`petfinder.exceptions.LimitExceeded` once
            you have reached the maximum number of records your credentials
            allow you to receive.
        """

        def pet_find_parser(root, has_records):
            """
            The parser that is used with the ``_do_autopaginating_api_call``
            method for auto-pagination.

            :param lxml.etree._Element root: The root Element in the response.
            :param dict has_records: A dict that we track the loop state in.
                dicts are passed by references, which is how this works.
            """
            for pet in root.findall("pets/pet"):
                # This is changed in the original record, since it's passed
                # by reference.
                has_records["has_records"] = True
                yield self._parse_pet_record(pet)

        return self._do_autopaginating_api_call(
            "pet.find", kwargs, pet_find_parser
        )