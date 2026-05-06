def existing_choice_control(self):
        """
        It controls errors. It generates an error message
        if zero or more than one channels are selected.
        """
        self.current.task_data['existing'] = False
        self.current.task_data['msg'] = _(u"You should choose just one channel to do operation.")
        keys, names = self.return_selected_form_items(self.input['form']['ChannelList'])
        if len(keys) == 1:
            self.current.task_data['existing'] = True
            self.current.task_data['target_channel_key'] = keys[0]