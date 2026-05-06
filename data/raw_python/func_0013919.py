def get_all(self, type_name, base_fields=None, the_filter=None,
                nested_fields=None):
        """Get the resource by resource id.

        :param nested_fields: nested resource fields
        :param base_fields: fields of this resource
        :param the_filter: dictionary of filter like `{'name': 'abc'}`
        :param type_name: Resource type. For example, pool, lun, nasServer.
        :return: List of resource class objects
        """
        fields = self.get_fields(type_name, base_fields, nested_fields)
        the_filter = self.dict_to_filter_string(the_filter)

        url = '/api/types/{}/instances'.format(type_name)

        resp = self.rest_get(url, fields=fields, filter=the_filter)
        ret = resp
        while resp.has_next_page:
            resp = self.rest_get(url, fields=fields, filter=the_filter,
                                 page=resp.next_page)
            ret.entries.extend(resp.entries)
        return ret