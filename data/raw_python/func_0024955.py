def patch_collection(self, collection, changes):
        """
        Will make specific updates to a record based on JSON Patch
        documentation.

            https://tools.ietf.org/html/rfc6902

        the format of changes is something like::

            [{
                'op': 'add',
                'path': '/newfield',
                'value': 'just added'
            }]

        """
        uri = str.join('/', [self.uri, collection])
        return self.service._patch(uri, changes)