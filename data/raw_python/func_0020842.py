def edit_form(self, obj):
        """Customize edit form."""
        form = super(OAISetModelView, self).edit_form(obj)
        del form.spec
        return form