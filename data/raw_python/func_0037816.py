def _create(self):
        """ Create new object on IxNetwork.

        :return: IXN object reference.
        """

        if 'name' in self._data:
            obj_ref = self.api.add(self.obj_parent(), self.obj_type(), name=self.obj_name())
        else:
            obj_ref = self.api.add(self.obj_parent(), self.obj_type())
        self.api.commit()
        return self.api.remapIds(obj_ref)