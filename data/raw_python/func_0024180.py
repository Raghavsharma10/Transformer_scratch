def selection_error_control(self, form_info):
        """
        It controls the selection from the form according
        to the operations, and returns an error message
        if it does not comply with the rules.

        Args:
            form_info: Channel or subscriber form from the user

        Returns: True or False
                 error message

        """
        keys, names = self.return_selected_form_items(form_info['ChannelList'])
        chosen_channels_number = len(keys)

        if form_info['new_channel'] and chosen_channels_number < 2:
            return False, _(
                u"You should choose at least two channel to merge operation at a new channel.")
        elif form_info['existing_channel'] and chosen_channels_number == 0:
            return False, _(
                u"You should choose at least one channel to merge operation with existing channel.")
        elif form_info['find_chosen_channel'] and chosen_channels_number != 1:
            return False, _(u"You should choose one channel for split operation.")

        return True, None