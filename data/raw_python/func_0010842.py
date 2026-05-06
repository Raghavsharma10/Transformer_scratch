def get_form_kwargs(self):
        """
        We override this, using only those fields specified if they are specified.

        Otherwise we include all fields in a standard ModelForm.
        """
        kwargs = super(SmartFormMixin, self).get_form_kwargs()
        kwargs['initial'] = self.derive_initial()
        return kwargs