def _ask_which_gist(self, matches):
        """
        Asks user which gist to use in case of more than one gist matching the
        instance filename.
        :param matches: (list) of dictioaries generated within select_gists()
        :return: (dict) of the selected gist
        """
        # ask user which gist to use
        self.hey("Use {} from which gist?".format(self.filename))
        for count, gist in enumerate(matches, 1):
            self.hey("[{}] {}".format(count, gist.get("description")))

        # get the gist index
        selected = False
        while not selected:
            gist_index = prompt("Type the number: ", type=int) - 1
            try:
                selected = matches[gist_index]
            except IndexError:
                self.oops("Invalid number, please try again.")

        self.output("Using `{}` Gist".format(selected["description"]))
        return selected