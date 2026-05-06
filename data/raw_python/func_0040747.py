def __on_message(self, connection, sender, message):
        """
        Got a message from a channel
        """
        if message.strip() == '!bot send':
            cycle = itertools.cycle(string.digits)
            content = ''.join(next(cycle) for _ in range(100))
            self.send_message(sender, content)

        else:
            parts = message.split(':', 2)
            if not parts or parts[0] != 'HRLD':
                return

            if parts[1] == 'BEGIN':
                # Beginning of multi-line message
                self._queue[parts[2]] = []

            elif parts[1] == 'END':
                # End of multi-line message
                content = ''.join(self._queue.pop(parts[2]))
                self.__notify(sender, content)

            elif parts[1] == 'MSG':
                # Single-line message
                content = parts[2]
                self.__notify(sender, content)

            else:
                # Multi-line message continuation
                uid = parts[1]
                self._queue[uid].append(parts[2])