def edit(self):
        """
        Edit file with default os application.
        """
        if platform.system().lower() == 'windows':
            os.startfile(str(self.config_file))
        else:
            if platform.system().lower() == 'darwin':
                call = 'open'
            else:
                call = 'xdg-open'
            subprocess.call([call, self.config_file])