def load(self, revision_path):
        """
        Load revision file.

        :param revision_path:
        :type revision_path: str
        """
        if not os.path.exists(revision_path):
            raise RuntimeError("revision file does not exist.")

        with open(revision_path, mode='r') as f:
            text = f.read()
            rev_strings = text.split("## ")

            for rev_string in rev_strings:
                if len(rev_string) == 0 or rev_string[:2] == "# ":
                    continue

                try:
                    revision = Revision()
                    revision.parse(rev_string)
                except RuntimeError:
                    raise RuntimeError("")

                self.insert(revision, len(self.revisions))