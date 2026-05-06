def on_message(self, sender, content):
        """
        Got a message from the client
        """
        try:
            message = json.loads(content)

        except (ValueError, TypeError) as ex:
            logging.error("Not a valid JSON string: %s", ex)
            return

        try:
            # Check the replied message
            reply_uid = message['reply-to']
            reply_level = message['reply-level']

        except KeyError:
            # Got a new message
            logging.info("Got message %s from %s", message['content'], sender)

            # Notify listeners
            self.__pool.enqueue(self._notify_listeners, sender, message)

        else:
            # Got a reply
            try:
                level, callback = self.__callbacks[reply_uid]

            except KeyError:
                # Nobody to callback...
                pass

            else:
                if level == reply_level:
                    # Match
                    try:
                        callback(sender, message['payload'])
                    except Exception as ex:
                        logging.exception("Error notifying sender: %s", ex)