def get_crud_menus(self):
        """
        Generates menu entries according to
        :attr:`zengine.settings.OBJECT_MENU` and permissions
        of current user.

        Returns:
            Dict of list of dicts (``{'':[{}],}``). Menu entries.
        """
        results = defaultdict(list)
        for object_type in settings.OBJECT_MENU:
            for model_data in settings.OBJECT_MENU[object_type]:
                if self.current.has_permission(model_data.get('wf', model_data['name'])):
                    self._add_crud(model_data, object_type, results)
        return results