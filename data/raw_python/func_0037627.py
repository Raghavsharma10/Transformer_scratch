def get_form(self, step=None, data=None, files=None):
        """Instanciate the form for the current step

        FormAdminView from xadmin expects form to be at self.form_obj

        """
        self.form_obj = super(FormWizardAdminView, self).get_form(
            step=step, data=data, files=files)
        return self.form_obj