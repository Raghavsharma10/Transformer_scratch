def format_messages(self, messages):
        """ Formats several messages with :class:Look, encodes them
            with :func:vital.tools.encoding.stdout_encode """
        mess = ""
        for message in self.message:
            if self.pretty:
                mess = "{}{}".format(mess, self.format_message(message))
            else:
                mess += str(message)
        if self.include_time:
            return ": {} : {}".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mess)
        return stdout_encode(mess)