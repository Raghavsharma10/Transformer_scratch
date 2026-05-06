def from_desmond(cls, path, **kwargs):
        """
        Loads a topology from a Desmond DMS file located at `path`.

        Arguments
        ---------
        path : str
            Path to a Desmond DMS file
        """
        dms = DesmondDMSFile(path)
        pos = kwargs.pop('positions', dms.getPositions())
        return cls(master=dms, topology=dms.getTopology(), positions=pos, path=path,
                   **kwargs)