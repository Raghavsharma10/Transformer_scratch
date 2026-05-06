def select_gist(self, allow_none=False):
        """
        Given the requested filename, it selects the proper gist; if more than
        one gist is found with the given filename, user is asked to choose.
        :allow_none: (bool) for `getgist` it should raise error if no gist is
        found, but setting this argument to True avoid this error, which is
        useful when `putgist` is calling this method
        :return: (dict) selected gist
        """
        # pick up all macthing gists
        matches = list()
        for gist in self.get_gists():
            for gist_file in gist.get("files"):
                if self.filename == gist_file.get("filename"):
                    matches.append(gist)

        # abort if no match is found
        if not matches:
            if allow_none:
                return None
            else:
                msg = "No file named `{}` found in {}'s gists"
                self.oops(msg.format(self.file_path, self.user))
                if not self.is_authenticated:
                    self.warn("To access private gists set the GETGIST_TOKEN")
                    self.warn("(see `getgist --help` for details)")
                return False

        # return if there's is only one match
        if len(matches) == 1 or self.assume_yes:
            return matches.pop(0)

        return self._ask_which_gist(matches)