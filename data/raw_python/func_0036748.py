def shelter_find(self, **kwargs):
        """
        shelter.find wrapper. Returns a generator of shelter record dicts
        matching your search criteria.

        :rtype: generator
        :returns: A generator of shelter record dicts.
        :raises: :py:exc:`petfinder.exceptions.LimitExceeded` once you have
            reached the maximum number of records your credentials allow you
            to receive.
        """

        def shelter_find_parser(root, has_records):
            """
            The parser that is used with the ``_do_autopaginating_api_call``
            method for auto-pagination.

            :param lxml.etree._Element root: The root Element in the response.
            :param dict has_records: A dict that we track the loop state in.
                dicts are passed by references, which is how this works.
            """
            for shelter in root.find("shelters"):
                has_records["has_records"] = True
                record = {}
                for field in shelter:
                    record[field.tag] = field.text
                yield record

        return self._do_autopaginating_api_call(
            "shelter.find", kwargs, shelter_find_parser
        )