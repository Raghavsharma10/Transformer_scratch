def calc_lfp(self):
        """ Sum all the LFP contributions from every cell type.
        """

        LFParray = np.array([])
        LFPdict = {}

        i = 0
        for y in self.y:
            fil = os.path.join(self.populations_path,
                               self.output_file.format(y, 'LFP.h5'))

            f = h5py.File(fil)

            if i == 0:
                LFParray = np.zeros((len(self.y),
                                    f['data'].shape[0], f['data'].shape[1]))

            #fill in
            LFParray[i, ] = f['data'].value

            LFPdict.update({y : f['data'].value})

            f.close()

            i += 1

        return LFPdict,  LFParray.sum(axis=0)