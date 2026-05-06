def edit(self, name, label=None):
        """Edit this asset.

        :param str name: (required), The file name of the asset
        :param str label: (optional), An alternate description of the asset
        :returns: boolean
        """
        if not name:
            return False
        edit_data = {'name': name, 'label': label}
        self._remove_none(edit_data)
        r = self._patch(
            self._api,
            data=json.dumps(edit_data),
            headers=Release.CUSTOM_HEADERS
        )
        successful = self._boolean(r, 200, 404)
        if successful:
            self.__init__(r.json(), self)

        return successful