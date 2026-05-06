def add_resource(self, resource_id, attributes, parents=[],
            issuer='default'):
        """
        Will add the given resource with a given identifier and attribute
        dictionary.

            example/

                add_resource('/asset/12', {'id': 12, 'manufacturer': 'GE'})
        """
        # MAINT: consider test to avoid adding duplicate resource id
        assert isinstance(attributes, (dict)), "attributes expected to be dict"

        attrs = []
        for key in attributes.keys():
            attrs.append({
                'issuer': issuer,
                'name': key,
                'value': attributes[key]
                })

        body = {
            "resourceIdentifier": resource_id,
            "parents": parents,
            "attributes": attrs,
        }

        return self._put_resource(resource_id, body)