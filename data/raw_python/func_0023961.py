def get_form(self, request, obj=None, **kwargs):
        """Change the form depending on whether we're adding or
           editing the slot."""
        if obj is None:
            # Adding a new Slot
            kwargs['form'] = SlotAdminAddForm
        return super(SlotAdmin, self).get_form(request, obj, **kwargs)