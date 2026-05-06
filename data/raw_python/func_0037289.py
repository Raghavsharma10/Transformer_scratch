def do_ls(self, nothing = ''):
        """list files in current remote directory"""
        for d in self.dirs:
            self.stdout.write("\033[0;34m" + ('%s\n' % d) + "\033[0m")

        for f in self.files:
            self.stdout.write('%s\n' % f)