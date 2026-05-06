def _read(self, fd, mask):
        """Read waiting data and terminate Tk mainloop if done"""
        try:
            # if EOF was encountered on a tty, avoid reading again because
            # it actually requests more data
            if select.select([fd],[],[],0)[0]:
                snew = os.read(fd, self.nbytes) # returns bytes in PY3K
                if PY3K: snew = snew.decode('ascii','replace')
                self.value.append(snew)
                self.nbytes -= len(snew)
            else:
                snew = ''
            if (self.nbytes <= 0 or len(snew) == 0) and self.widget:
                # stop the mainloop
                self.widget.quit()
        except OSError:
            raise IOError("Error reading from %s" % (fd,))