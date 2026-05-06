def channel_choice_control(self):
        """
        It controls errors. If there is an error,
        returns channel list screen with error message.
        """
        self.current.task_data['control'], self.current.task_data['msg'] \
            = self.selection_error_control(self.input['form'])
        if self.current.task_data['control']:
            self.current.task_data['option'] = self.input['cmd']
            self.current.task_data['split_operation'] = False
            keys, names = self.return_selected_form_items(self.input['form']['ChannelList'])
            self.current.task_data['chosen_channels'] = keys
            self.current.task_data['chosen_channels_names'] = names