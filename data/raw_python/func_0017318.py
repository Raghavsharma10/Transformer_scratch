def _write_commits_to_release_notes(self):
        """
        writes commits to the releasenotes file by appending to the end
        """
        with open(self.release_file, 'a') as out:
            out.write("==========\n{}\n".format(self.tag))
            for commit in self.commits:
                try:
                    msg = commit[1]
                    if msg != "cosmetic":
                        out.write("-" + msg + "\n")
                except:
                    pass