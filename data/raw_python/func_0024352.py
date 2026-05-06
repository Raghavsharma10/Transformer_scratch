def select_postponed_date(self):
        """
            The time intervals at which the workflow is to be extended are determined.
            .. code-block:: python

                #  request:
                   {
                   'task_inv_key': string,
                   }

        """

        _form = forms.JsonForm(title="Postponed Workflow")
        _form.start_date = fields.DateTime("Start Date")
        _form.finish_date = fields.DateTime("Finish Date")
        _form.save_button = fields.Button("Save")
        self.form_out(_form)