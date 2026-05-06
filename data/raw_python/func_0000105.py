def get_formset(self, request, obj=None, **kwargs):
        """ Default user to the current version owner. """
        data = super().get_formset(request, obj, **kwargs)
        if obj:
            data.form.base_fields['user'].initial = request.user.id
        return data