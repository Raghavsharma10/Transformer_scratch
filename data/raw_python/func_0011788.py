def load_nicknames(self, file):
        """
        Load dict from file for random nicknames.

        :param str file: filename
        """
        with open(os.path.join(main_dir, file + '.dat'), 'r') as f:
            self.nicknames = json.load(f)