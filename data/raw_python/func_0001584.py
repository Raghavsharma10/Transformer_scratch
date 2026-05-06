def _reload(self):
        """
        Gets every registered form's field value.\
        If a field name is found in the db, it will load it from there.\
        Otherwise, the initial value from the field form is used
        """
        ConfigModel = apps.get_model('djconfig.Config')
        cache = {}
        data = dict(
            ConfigModel.objects
                .all()
                .values_list('key', 'value'))

        # populate cache with initial form values,
        # then with cleaned database values,
        # then with raw database file/image paths
        for form_class in self._registry:
            empty_form = form_class()
            cache.update({
                name: field.initial
                for name, field in empty_form.fields.items()})
            form = form_class(data={
                name: _deserialize(data[name], field)
                for name, field in empty_form.fields.items()
                if name in data and not isinstance(field, forms.FileField)})
            form.is_valid()
            cache.update({
                name: _unlazify(value)
                for name, value in form.cleaned_data.items()
                if name in data})
            # files are special because they don't have an initial value
            # and the POSTED data must contain the file. So, we keep
            # the stored path as is
            # TODO: see if serialize/deserialize/unlazify can be used for this instead
            cache.update({
                name: data[name]
                for name, field in empty_form.fields.items()
                if name in data and isinstance(field, forms.FileField)})

        cache['_updated_at'] = data.get('_updated_at')
        self._cache = cache