def bot_watcher(self):
        """
        Thread (greenlet) that will try and reconnect the bot if
        it's not connected.
        """
        default_interval = 5
        interval = default_interval
        while True:
            if not self.bot.connection.connected:
                if self.bot.reconnect():
                    interval = default_interval
                else:
                    interval *= 2
            sleep(interval)