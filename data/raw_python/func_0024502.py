def list_user_roles(self):
        """
        Lists user roles as selectable except user's current role.
        """
        _form = JsonForm(current=self.current, title=_(u"Switch Role"))
        _form.help_text = "Your current role: %s %s" % (self.current.role.unit.name,
                                                        self.current.role.abstract_role.name)
        switch_roles = self.get_user_switchable_roles()
        _form.role_options = fields.Integer(_(u"Please, choose the role you want to switch:")
                                            , choices=switch_roles, default=switch_roles[0][0],
                                            required=True)
        _form.switch = fields.Button(_(u"Switch"))
        self.form_out(_form)