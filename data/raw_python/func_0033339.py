def mark_as_read(self, room_name):
        """
        message_ids return an array
        with unread message ids ['131313231', '323131']
        """
        api_meth = self.set_user_items_url(room_name)
        message_ids = self.unread_items(room_name).get('chat')
        if message_ids:
            return self.post(api_meth, data={'chat': message_ids})
        else:
            raise GitterItemsError(room_name)