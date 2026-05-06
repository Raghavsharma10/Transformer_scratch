def save_data(self, idx):
        """Save the internal data of all sequences with an activated flag.
        Write to file if the corresponding disk flag is activated; store
        in working memory if the corresponding ram flag is activated."""
        for name in self:
            actual = getattr(self, name)
            diskflag = getattr(self, '_%s_diskflag' % name)
            ramflag = getattr(self, '_%s_ramflag' % name)
            if diskflag:
                file_ = getattr(self, '_%s_file' % name)
                ndim = getattr(self, '_%s_ndim' % name)
                length_tot = 1
                for jdx in range(ndim):
                    length = getattr(self, '_%s_length_%s' % (name, jdx))
                    length_tot *= length
                if ndim:
                    raw = struct.pack(length_tot*'d', *actual.flatten())
                else:
                    raw = struct.pack('d', actual)
                file_.write(raw)
            elif ramflag:
                array = getattr(self, '_%s_array' % name)
                array[idx] = actual