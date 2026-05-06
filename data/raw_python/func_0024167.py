def channel_list(self):
        """
        Main screen for channel management.
        Channels listed and operations can be chosen on the screen.
        If there is an error message like non-choice,
        it is shown here.

        """

        if self.current.task_data.get('msg', False):
            if self.current.task_data.get('target_channel_key', False):
                self.current.output['msgbox'] = {'type': 'info',
                                                 "title": _(u"Successful Operation"),
                                                 "msg": self.current.task_data['msg']}
                del self.current.task_data['msg']
            else:
                self.show_warning_messages()

        self.current.task_data['new_channel'] = False
        _form = ChannelListForm(title=_(u'Public Channel List'), help_text=CHANNEL_CHOICE_TEXT)

        for channel in Channel.objects.filter(typ=15):
            owner_name = channel.owner.username
            _form.ChannelList(choice=False, name=channel.name, owner=owner_name, key=channel.key)

        _form.new_channel = fields.Button(_(u"Merge At New Channel"), cmd="create_new_channel")
        _form.existing_channel = fields.Button(_(u"Merge With An Existing Channel"),
                                               cmd="choose_existing_channel")
        _form.find_chosen_channel = fields.Button(_(u"Split Channel"), cmd="find_chosen_channel")
        self.form_out(_form)