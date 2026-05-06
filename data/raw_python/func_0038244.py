def _create(self, **attributes):
        """ Create new interface on IxNetwork.

        Set enabled and description (==name).

        :return: interface object reference.
        """

        attributes['enabled'] = True
        if 'name' in self._data:
            attributes['description'] = self._data['name']
        obj_ref = self.api.add(self.obj_parent(), self.obj_type(), **attributes)
        self.api.commit()
        return self.api.remapIds(obj_ref)