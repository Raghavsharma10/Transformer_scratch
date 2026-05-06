def save_hdf(self,filename,path='',res=1000,logspace=False):
        """Saves distribution to an HDF5 file.

        Saves a pandas `dataframe` object containing tabulated pdf and cdf
        values at a specfied resolution.  After saving to a particular path, a
        distribution may be regenerated using the `Distribution_FromH5` subclass.  

        Parameters
        ----------
        filename : string
            File in which to save the distribution.  Should end in .h5.

        path : string, optional
            Path in which to save the distribution within the .h5 file.  By
            default this is an empty string, which will lead to saving the
            `fns` dataframe at the root level of the file.

        res : int, optional
            Resolution at which to grid the distribution for saving.

        logspace : bool, optional
            Sets whether the tabulated function should be gridded with log or
            linear spacing.  Default will be logspace=False, corresponding
            to linear gridding.

        """
        if logspace:
            vals = np.logspace(np.log10(self.minval),
                               np.log10(self.maxval),
                               res)
        else:
            vals = np.linspace(self.minval,self.maxval,res)
        d = {'vals':vals,
             'pdf':self(vals),
             'cdf':self.cdf(vals)}
        df = pd.DataFrame(d)
        df.to_hdf(filename,path+'/fns')
        if hasattr(self,'samples'):
            s = pd.Series(self.samples)
            s.to_hdf(filename,path+'/samples')
        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/fns'.format(path)).attrs
        attrs.keywords = self.keywords
        attrs.disttype = type(self)
        store.close()