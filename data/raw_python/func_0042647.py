def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests.
        Validates submitted form and
        formsets. Saves if valid, re displays
        page with errors if invalid.
        """

        self.object = self.get_object()
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        formsets = self.get_formsets(form, saving=True)

        valid_formsets = True
        for formset in formsets.values():
            if not formset.is_valid():
                valid_formsets = False
                break

        if self.is_valid(form, formsets):
            return self.form_valid(form, formsets)
        else:
            adminForm = self.get_admin_form(form)
            adminFormSets = self.get_admin_formsets(formsets)
            context = {
                'adminForm': adminForm,
                'formsets': adminFormSets,
                'obj': self.object,
            }
            return self.form_invalid(form=form, **context)