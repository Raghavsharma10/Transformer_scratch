def save_new_channel(self):
        """
        It saves new channel according to specified channel features.

        """
        form_info = self.input['form']
        channel = Channel(typ=15, name=form_info['name'],
                          description=form_info['description'],
                          owner_id=form_info['owner_id'])
        channel.blocking_save()
        self.current.task_data['target_channel_key'] = channel.key