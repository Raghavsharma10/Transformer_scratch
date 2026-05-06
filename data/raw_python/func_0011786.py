def load_nouns(self, file):
        """
        Load dict from file for random words.

        :param str file: filename
        """
        with open(os.path.join(main_dir, file + '.dat'), 'r') as f:
            self.nouns = json.load(f)