def create_new_channel(self):
        """
        Features of new channel are specified like channel's name, owner etc.
        """

        self.current.task_data['new_channel'] = True
        _form = NewChannelForm(Channel(), current=self.current)
        _form.title = _(u"Specify Features of New Channel to Create")
        _form.forward = fields.Button(_(u"Create"), flow="find_target_channel")
        self.form_out(_form)