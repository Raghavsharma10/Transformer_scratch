def run(self):
        """
        run method
        """
        if self.pre_message is None:
            try:
                input()  # pre message of None causes the input to be silent
            except EOFError:
                pass
        else:
            try:
                input(self.pre_message)
            except EOFError:
                pass
        if self.post_message:
            print(self.post_message)
        self.exit_time = time.time()