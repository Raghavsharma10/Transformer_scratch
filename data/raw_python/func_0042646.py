def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests.
        Calls the `render` method with the following
        items in context.

        * **adminForm** - The main form wrapped in an helper class \
        that helps with fieldset iteration and html attributes.
        * **obj** - The object being edited.
        * **formsets** - Any attached formsets.
        """

        self.object = self.get_object()
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        formsets = self.get_formsets(form)

        adminForm = self.get_admin_form(form)
        adminFormSets = self.get_admin_formsets(formsets)
        context = {
            'adminForm': adminForm,
            'obj': self.object,
            'formsets': adminFormSets,
        }
        return self.render(request, **context)