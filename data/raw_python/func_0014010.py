def from_csv(cls, name=None, col_names=None, sep=None, **kwargs):
        """Create instrument metadata object from csv.

        Parameters
        ----------
        name : string
            absolute filename for csv file or name of file
            stored in pandas instruments location
        col_names : list-like collection of strings
            column names in csv and resultant meta object
        sep : string
            column seperator for supplied csv filename

        Note
        ----
        column names must include at least ['name', 'long_name', 'units'], 
        assumed if col_names is None.
        """
        import pysat
        req_names = ['name','long_name','units']
        if col_names is None:
            col_names = req_names
        elif not all([i in col_names for i in req_names]):
            raise ValueError('col_names must include name, long_name, units.')

        if sep is None:
            sep = ','

        if name is None:
            raise ValueError('Must supply an instrument name or file path.')
        elif not isinstance(name, str):
            raise ValueError('keyword name must be related to a string')
        elif not os.path.isfile(name):
                # Not a real file, assume input is a pysat instrument name
                # and look in the standard pysat location.
                test =  os.path.join(pysat.__path__[0],'instruments',name)
                if os.path.isfile(test):
                    name = test
                else:
                    #trying to form an absolute path for success
                    test = os.path.abspath(name)
                    if not os.path.isfile(test):
                        raise ValueError("Unable to create valid file path.")
                    else:
                        #success
                        name = test

        mdata = pds.read_csv(name, names=col_names, sep=sep, **kwargs) 

        if not mdata.empty:
            # make sure the data name is the index
            mdata.index = mdata['name']
            del mdata['name']
            return cls(metadata=mdata)
        else:
            raise ValueError('Unable to retrieve information from ' + name)