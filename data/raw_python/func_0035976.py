def _ions(self, f):
        """
        This is a generator that returns the mzs being measured during
        each time segment, one segment at a time.
        """
        outside_pos = f.tell()
        doff = find_offset(f, 4 * b'\xff' + 'HapsSearch'.encode('ascii'))
        # actual end of prev section is 34 bytes before, but assume 1 rec
        f.seek(doff - 62)
        # seek backwards to find the FFFFFFFF header
        while True:
            f.seek(f.tell() - 8)
            if f.read(4) == 4 * b'\xff':
                break
        f.seek(f.tell() + 64)
        nsegments = struct.unpack('<I', f.read(4))[0]
        for _ in range(nsegments):
            # first 32 bytes are segment name, rest are something else?
            f.seek(f.tell() + 96)
            nions = struct.unpack('<I', f.read(4))[0]
            ions = []
            for _ in range(nions):
                # TODO: check that itype is actually a SIM/full scan switch
                i1, i2, _, _, _, _, itype, _ = struct.unpack('<' + 8 * 'I',
                                                             f.read(32))
                if itype == 0:  # SIM
                    ions.append(i1 / 100.)
                else:  # full scan
                    # TODO: this might be a little hacky?
                    #  ideally we would need to know n for this, e.g.:
                    # ions += np.linspace(i1 / 100, i2 / 100, n).tolist()
                    ions += np.arange(i1 / 100., i2 / 100. + 1, 1).tolist()
            # save the file position and load the position
            # that we were at before we started this code
            inside_pos = f.tell()
            f.seek(outside_pos)
            yield ions
            outside_pos = f.tell()
            f.seek(inside_pos)
        f.seek(outside_pos)