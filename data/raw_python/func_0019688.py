def save(self, fname, compression='blosc'):
        """
        Save method for the Egg object

        The data will be saved as a 'egg' file, which is a dictionary containing
        the elements of a Egg saved in the hd5 format using
        `deepdish`.

        Parameters
        ----------

        fname : str
            A name for the file.  If the file extension (.egg) is not specified,
            it will be appended.

        compression : str
            The kind of compression to use.  See the deepdish documentation for
            options: http://deepdish.readthedocs.io/en/latest/api_io.html#deepdish.io.save

        """

        # put egg vars into a dict
        egg = {
            'pres' : df2list(self.pres),
            'rec' : df2list(self.rec),
            'dist_funcs' : self.dist_funcs,
            'subjgroup' : self.subjgroup,
            'subjname' : self.subjname,
            'listgroup' : self.listgroup,
            'listname' : self.listname,
            'date_created' : self.date_created,
            'meta' : self.meta
        }

        # if extension wasn't included, add it
        if fname[-4:]!='.egg':
            fname+='.egg'

        # save
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd.io.save(fname, egg, compression=compression)