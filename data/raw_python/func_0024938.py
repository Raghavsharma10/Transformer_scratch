def delete_space(self, name=None, guid=None):
        """
        Delete the current space, or a space with the given name
        or guid.
        """

        if not guid:
            if name:
                spaces = self._get_spaces()
                for space in spaces['resources']:
                    if space['entity']['name'] == name:
                        guid = space['metadata']['guid']
                        break
                if not guid:
                    raise ValueError("Space with name %s not found." % (name))
            else:
                guid = self.guid

        logging.warning("Deleting space (%s) and all services." % (guid))

        return self.api.delete("/v2/spaces/%s" % (guid), params={'recursive':
        'true'})