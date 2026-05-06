def get(self, type_name, obj_id, base_fields=None, nested_fields=None):
        """Get the resource by resource id.

        :param nested_fields: nested resource fields.
        :param type_name: Resource type. For example, pool, lun, nasServer.
        :param obj_id: Resource id
        :param base_fields: Resource fields to return
        :return: List of tuple [(name, res_inst)]
        """
        base_fields = self.get_fields(type_name, base_fields, nested_fields)
        url = '/api/instances/{}/{}'.format(type_name, obj_id)
        return self.rest_get(url, fields=base_fields)