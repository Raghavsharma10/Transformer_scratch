def _blockread(self, fname):
        """
        Generator yields bsize lines from gdf file.
        Hidden method.


        Parameters
        ----------
        fname : str
            Name of gdf-file.
            
        
        Yields
        ------
        list
            file contents
            
        """
        with open(fname, 'rb') as f:
            while True:
                a = []
                for i in range(self.bsize):
                    line = f.readline()
                    if not line: break
                    a.append(line.split())
                if a == []: raise StopIteration
                yield a