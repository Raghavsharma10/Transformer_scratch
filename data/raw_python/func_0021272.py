def open_magnet(self):
        """Open magnet according to os."""
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', self.magnet],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif sys.platform.startswith('win32'):
            os.startfile(self.magnet)
        elif sys.platform.startswith('cygwin'):
            os.startfile(self.magnet)
        elif sys.platform.startswith('darwin'):
            subprocess.Popen(['open', self.magnet],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            subprocess.Popen(['xdg-open', self.magnet],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)