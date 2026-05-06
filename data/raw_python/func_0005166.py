def format_message(self, message):
        """ Formats a message with :class:Look """
        look = Look(message)
        return look.pretty(display=False)