def read(self, file, nbytes):
        """Read nbytes characters from file while running Tk mainloop"""
        if not capable.OF_GRAPHICS:
            raise RuntimeError("Cannot run this command without graphics")
        if isinstance(file, int):
            fd = file
        else:
            # Otherwise, assume we have Python file object
            try:
                fd = file.fileno()

            except:
                raise TypeError("file must be an integer or a filehandle/socket")
        init_tk_default_root() # harmless if already done
        self.widget = TKNTR._default_root
        if not self.widget:
            # no Tk widgets yet, so no need for mainloop
            # (shouldnt happen now with init_tk_default_root)
            s = []
            while nbytes>0:
                snew = os.read(fd, nbytes) # returns bytes in PY3K
                if snew:
                    if PY3K: snew = snew.decode('ascii','replace')
                    s.append(snew)
                    nbytes -= len(snew)
                else:
                    # EOF -- just return what we have so far
                    break
            return "".join(s)
        else:
            self.nbytes = nbytes
            self.value = []
            self.widget.tk.createfilehandler(fd,
                                    TKNTR.READABLE | TKNTR.EXCEPTION,
                                    self._read)
            try:
                self.widget.mainloop()
            finally:
                self.widget.tk.deletefilehandler(fd)
            return "".join(self.value)