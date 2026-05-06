def setupNodding(self):
        """
        Setup Nodding for GTC
        """
        g = get_root(self).globals

        if not self.nod():
            # re-enable clear mode box if not drift
            if not self.isDrift():
                self.clear.enable()

            # clear existing nod pattern
            self.nodPattern = {}
            self.check()
            return

        # Do nothing if we're not at the GTC
        if g.cpars['telins_name'] != 'GTC':
            messagebox.showerror('Error', 'Cannot dither WHT')
            self.nod.set(False)
            self.nodPattern = {}
            return

        # check for drift mode and bomb out
        if self.isDrift():
            messagebox.showerror('Error', 'Cannot dither telescope in drift mode')
            self.nod.set(False)
            self.nodPattern = {}
            return

        # check for clear not enabled and warn
        if not self.clear():
            if not messagebox.askokcancel('Warning',
                                          'Dithering telescope will enable clear mode. Continue?'):
                self.nod.set(False)
                self.nodPattern = {}
                return

        # Ask for nod pattern
        try:
            home = expanduser('~')
            fname = filedialog.askopenfilename(
                title='Open offsets text file',
                defaultextension='.txt',
                filetypes=[('text files', '.txt')],
                initialdir=home)

            if not fname:
                g.clog.warn('Aborted load from disk')
                raise ValueError

            ra, dec = np.loadtxt(fname).T
            if len(ra) != len(dec):
                g.clog.warn('Mismatched lengths of RA and Dec offsets')
                raise ValueError

            data = dict(
                ra=ra.tolist(),
                dec=dec.tolist()
            )
        except:
            g.clog.warn('Setting dither pattern failed. Disabling dithering')
            self.nod.set(False)
            self.nodPattern = {}
            return

        # store nodding on ipars object
        self.nodPattern = data
        # enable clear mode
        self.clear.set(True)
        # update
        self.check()