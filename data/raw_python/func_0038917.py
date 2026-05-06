def on_message(self, message_id_service, contact_id_service, content):
        """
        To use as callback in message service backend
        """
        try:
            live_chat = Chat.live.get(
                Q(agent__id_service=contact_id_service) | Q(asker__id_service=contact_id_service))            
        except ObjectDoesNotExist:
            self._new_chat_processing(message_id_service, contact_id_service, content)
        else:
            live_chat.handle_message(message_id_service, contact_id_service, content, self)