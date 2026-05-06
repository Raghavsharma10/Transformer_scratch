def show(self):
        """Show the overfitting PDF summary."""
        try:
            if platform.system().lower().startswith('darwin'):
                subprocess.call(['open', self.pdf])
            elif os.name == 'nt':
                os.startfile(self.pdf)
            elif os.name == 'posix':
                subprocess.call(['xdg-open', self.pdf])
            else:
                raise IOError("")
        except IOError:
            log.info("Unable to open the pdf. Try opening it manually:")
            log.info(self.pdf)