def select_role(self):
        """
        The workflow method to be assigned to the person with the same role and unit as the user.
            .. code-block:: python

                #  request:
                   {
                   'task_inv_key': string,
                   }

        """

        roles = [(m.key, m.__unicode__()) for m in RoleModel.objects.filter(
            abstract_role=self.current.role.abstract_role,
            unit=self.current.role.unit) if m != self.current.role]

        if roles:
            _form = forms.JsonForm(title=_(u'Assign to workflow'))
            _form.select_role = fields.Integer(_(u"Chose Role"), choices=roles)
            _form.explain_text = fields.String(_(u"Explain Text"), required=False)
            _form.send_button = fields.Button(_(u"Send"))
            self.form_out(_form)
        else:
            title = _(u"Unsuccessful")
            msg = _(u"Assign role not found")
            self.current.msg_box(title=title, msg=msg)