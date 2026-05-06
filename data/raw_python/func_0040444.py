def load(self,dset):
        '''load a dataset from given filename into the object'''
        self.dset_filename = dset
        self.dset = nib.load(dset)
        self.data = self.dset.get_data()
        self.header = self.dset.get_header()