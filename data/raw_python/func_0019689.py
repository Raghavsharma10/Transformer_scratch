def save(self, fname, compression='blosc'):
        """
        Save method for the FriedEgg object

        The data will be saved as a 'fegg' file, which is a dictionary containing
        the elements of a FriedEgg saved in the hd5 format using
        `deepdish`.

        Parameters
        ----------

        fname : str
            A name for the file.  If the file extension (.fegg) is not specified,
            it will be appended.

        compression : str
            The kind of compression to use.  See the deepdish documentation for
            options: http://deepdish.readthedocs.io/en/latest/api_io.html#deepdish.io.save

        """

        egg = {
            'data' : self.data,
            'analysis' : self.analysis,
            'list_length' : self.list_length,
            'n_lists' : self.n_lists,
            'n_subjects' : self.n_subjects,
            'position' : self.position,
            'date_created' : self.date_created,
            'meta' : self.meta
        }

        if fname[-4:]!='.fegg':
            fname+='.fegg'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd.io.save(fname, egg, compression=compression)