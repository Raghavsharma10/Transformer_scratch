def _print_message(self, flag_message=None, color=None, padding=None,
                       reverse=False):
        """ Outputs the message to the terminal """
        if flag_message:
            flag_message = stdout_encode(flag(flag_message,
                                              color=color if self.pretty else None,
                                              show=False))
            if not reverse:
                print(padd(flag_message, padding),
                      self.format_messages(self.message))
            else:
                print(self.format_messages(self.message),
                      padd(flag_message, padding))
        else:
            print(self.format_messages(self.message))
        self.message = []