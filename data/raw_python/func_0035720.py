def go(self, settings, command):
        """
        Run the specified command using a TwitterBot created with the provided settings
        :param settings: Settings class
        :param command: Command to run, either 'post_message' or 'reply_to_mentions'
        :return: Result of running the command
        """
        bot = TwitterBot(settings)

        result = 1
        if command == 'post_message':
            result = bot.post_message()
        elif command == 'reply_to_mentions':
            result = bot.reply_to_mentions()
        else:
            print("Command must be either 'post_message' or 'reply_to_mentions'")

        return result