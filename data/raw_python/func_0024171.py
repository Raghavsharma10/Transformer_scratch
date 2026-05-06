def choose_existing_channel(self):
        """
        It is a channel choice list and chosen channels
        at previous step shouldn't be on the screen.
        """

        if self.current.task_data.get('msg', False):
            self.show_warning_messages()

        _form = ChannelListForm()
        _form.title = _(u"Choose a Channel Which Will Be Merged With Chosen Channels")

        for channel in Channel.objects.filter(typ=15).exclude(
                key__in=self.current.task_data['chosen_channels']):
            owner_name = channel.owner.username
            _form.ChannelList(choice=False, name=channel.name, owner=owner_name,
                              key=channel.key)

        _form.choose = fields.Button(_(u"Choose"))
        self.form_out(_form)