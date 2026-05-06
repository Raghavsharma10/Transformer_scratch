def _add_crud(self, model_data, object_type, results):
        """
        Creates a menu entry for given model data.
        Updates results in place.

        Args:
            model_data: Model data.
            object_type: Relation name.
            results: Results dict.
        """
        model = model_registry.get_model(model_data['name'])
        field_name = model_data.get('field')
        verbose_name = model_data.get('verbose_name', model.Meta.verbose_name_plural)
        category = model_data.get('category', settings.DEFAULT_OBJECT_CATEGORY_NAME)
        wf_dict = {"text": verbose_name,
                   "wf": model_data.get('wf', "crud"),
                   "model": model_data['name'],
                   "kategori": category}
        if field_name:
            wf_dict['param'] = field_name
        results[object_type].append(wf_dict)
        self._add_to_quick_menu(wf_dict['model'], wf_dict)