def simple_crud():
        """
        Prepares menu entries for auto-generated model CRUD views.
        This is simple version of :attr:`get_crud_menus()` without
        Category support and permission control.
        Just for development purposes.

        Returns:
            Dict of list of dicts (``{'':[{}],}``). Menu entries.

        """
        results = defaultdict(list)
        for mdl in model_registry.get_base_models():
            results['other'].append({"text": mdl.Meta.verbose_name_plural,
                                     "wf": 'crud',
                                     "model": mdl.__name__,
                                     "kategori": settings.DEFAULT_OBJECT_CATEGORY_NAME})
        return results