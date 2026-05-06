def get_formset(self, request, obj=None, **kwargs):
        """
        Pre-populate formset with the initial placeholders to display.
        """
        def _placeholder_initial(p):
            # p.as_dict() returns allowed_plugins too for the client-side API.
            return {
                'slot': p.slot,
                'title': p.title,
                'role': p.role,
            }

        # Note this method is called twice, the second time in get_fieldsets() as `get_formset(request).form`
        initial = []
        if request.method == "GET":
            placeholder_admin = self._get_parent_modeladmin()

            # Grab the initial data from the parent PlaceholderEditorBaseMixin
            data = placeholder_admin.get_placeholder_data(request, obj)
            initial = [_placeholder_initial(d) for d in data]

            # Order initial properly,

        # Inject as default parameter to the constructor
        # This is the BaseExtendedGenericInlineFormSet constructor
        FormSetClass = super(PlaceholderEditorInline, self).get_formset(request, obj, **kwargs)
        FormSetClass.__init__ = curry(FormSetClass.__init__, initial=initial)
        return FormSetClass