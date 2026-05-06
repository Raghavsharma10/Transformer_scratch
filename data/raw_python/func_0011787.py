def load_dmails(self, file):
        """
        Load list from file for random mails

        :param str file: filename
        """
        with open(os.path.join(main_dir, file + '.dat'), 'r') as f:
            self.dmails = frozenset(json.load(f))