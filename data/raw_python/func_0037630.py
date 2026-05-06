def get_context(self, **kwargs):
        """Use this method to built context data for the template

        Mix django wizard context data with django-xadmin context

        """
        context = self.get_context_data(form=self.form_obj, **kwargs)
        context.update(super(FormAdminView, self).get_context())
        return context