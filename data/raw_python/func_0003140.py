def smartfields_get_field_status(self, field_name):
        """A way to find out a status of a filed."""
        manager = self._smartfields_managers.get(field_name, None)
        if manager is not None:
            return manager.get_status(self)
        return {'state': 'ready'}