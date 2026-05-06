def calc_csd(self):
        """ Sum all the CSD contributions from every layer.
        """

        CSDarray = np.array([])
        CSDdict = {}

        i = 0
        for y in self.y:
            fil = os.path.join(self.populations_path,
                               self.output_file.format(y, 'CSD.h5'))

            f = h5py.File(fil)

            if i == 0:
                CSDarray = np.zeros((len(self.y),
                                    f['data'].shape[0], f['data'].shape[1]))

            #fill in
            CSDarray[i, ] = f['data'].value

            CSDdict.update({y : f['data'].value})

            f.close()

            i += 1

        return CSDdict,  CSDarray.sum(axis=0)