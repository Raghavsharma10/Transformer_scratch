def shelter_listbybreed(self, **kwargs):
        """
        shelter.listByBreed wrapper. Given a breed and an animal type, list
        the shelter IDs with pets of said breed.

        :rtype: generator
        :returns: A generator of shelter IDs that have breed matches.
        """

        root = self._do_api_call("shelter.listByBreed", kwargs)

        shelter_ids = root.findall("shelterIds/id")
        for shelter_id in shelter_ids:
            yield shelter_id.text