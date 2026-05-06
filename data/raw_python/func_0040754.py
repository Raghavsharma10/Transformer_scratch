def _notify_listeners(self, sender, message):
        """
        Notifies listeners of a new message
        """
        uid = message['uid']
        msg_topic = message['topic']


        self._ack(sender, uid, 'fire')

        all_listeners = set()
        for lst_topic, listeners in self.__listeners.items():
            if fnmatch.fnmatch(msg_topic, lst_topic):
                all_listeners.update(listeners)

        self._ack(sender, uid, 'notice', 'ok' if all_listeners else 'none')

        try:
            results = []
            for listener in all_listeners:
                result = listener.handle_message(sender,
                                                 message['topic'],
                                                 message['content'])
                if result:
                    results.append(result)

            self._ack(sender, uid, 'send', json.dumps(results))
        except:
            self._ack(sender, uid, 'send', "Error")