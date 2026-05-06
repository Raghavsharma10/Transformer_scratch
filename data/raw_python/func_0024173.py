def split_channel(self):
        """
        A channel can be splitted to new channel or other existing channel.
        It creates subscribers list as selectable to moved.
        """

        if self.current.task_data.get('msg', False):
            self.show_warning_messages()

        self.current.task_data['split_operation'] = True
        channel = Channel.objects.get(self.current.task_data['chosen_channels'][0])

        _form = SubscriberListForm(title=_(u'Choose Subscribers to Migrate'))

        for subscriber in Subscriber.objects.filter(channel=channel):
            subscriber_name = subscriber.user.username
            _form.SubscriberList(choice=False, name=subscriber_name, key=subscriber.key)

        _form.new_channel = fields.Button(_(u"Move to a New Channel"), cmd="create_new_channel")
        _form.existing_channel = fields.Button(_(u"Move to an Existing Channel"),
                                               cmd="choose_existing_channel")
        self.form_out(_form)