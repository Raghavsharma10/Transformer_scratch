def readShiftFile(self, filename):
        """
        Reads a shift file from disk and populates a dictionary.
        """
        order = []
        fshift = open(filename,'r')
        flines = fshift.readlines()
        fshift.close()

        common = [f.strip('#').strip() for f in flines if f.startswith('#')]
        c=[line.split(': ') for line in common]

        # Remove any line comments in the shift file - lines starting with '#'
        # but not part of the common block.
        for l in c:
            if l[0] not in ['frame', 'refimage', 'form', 'units']:
                c.remove(l)

        for line in c: line[1]=line[1].strip()
        self.update(c)

        files = [f.strip().split(' ',1) for f in flines if not (f.startswith('#') or f.strip() == '')]
        for f in files:
            order.append(f[0])

        self['order'] = order

        for f in files:
            # Check to see if filename provided is a full filename that corresponds
            # to a file on the path.  If not, try to convert given rootname into
            # a valid filename based on available files.  This may or may not
            # define the correct filename, which is why it prints out what it is
            # doing, so that the user can verify and edit the shiftfile if needed.
            #NOTE:
            # Supporting the specification of only rootnames in the shiftfile with this
            # filename expansion is NOT to be documented, but provided solely as
            # an undocumented, dangerous and not fully supported helper function for
            # some backwards compatibility.
            if not os.path.exists(f[0]):
                f[0] = fu.buildRootname(f[0])
                print('Defining filename in shiftfile as: ', f[0])

            f[1] = f[1].split()
            try:
                f[1] = [float(s) for s in f[1]]
            except:
                msg = 'Cannot read in ', s, ' from shiftfile ', filename, ' as a float number'
                raise ValueError(msg)
            msg = "At least 2 and at most 4 shift values should be provided in a shiftfile"
            if len(f[1]) < 2:
                raise ValueError(msg)
            elif len(f[1]) == 3:
                f[1].append(1.0)
            elif len(f[1]) == 2:
                f[1].extend([0.0, 1.0])
            elif len(f[1]) > 4:
                raise ValueError(msg)

        fdict = dict(files)
        self.update(fdict)