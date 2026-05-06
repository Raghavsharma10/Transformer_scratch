def adjust_attributes_on_object(self, collection, name, things, values, how):
        """
        adjust labels or annotations on object

        labels have to match RE: (([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])? and
        have at most 63 chars

        :param collection: str, object collection e.g. 'builds'
        :param name: str, name of object
        :param things: str, 'labels' or 'annotations'
        :param values: dict, values to set
        :param how: callable, how to adjust the values e.g.
                    self._replace_metadata_things
        :return:
        """
        url = self._build_url("%s/%s" % (collection, name))
        response = self._get(url)
        logger.debug("before modification: %s", response.content)
        build_json = response.json()
        how(build_json['metadata'], things, values)
        response = self._put(url, data=json.dumps(build_json), use_json=True)
        check_response(response)
        return response