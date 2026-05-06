def create(self, re='brunel-py-ex-*.gdf', index=True):
        """
        Create db from list of gdf file glob


        Parameters
        ----------
        re : str
            File glob to load.
        index : bool
            Create index on neurons for speed.
                    
        
        Returns
        -------
        None
        
        
        See also
        --------
        sqlite3.connect.cursor, sqlite3.connect
        
        """
        self.cursor.execute('CREATE TABLE IF NOT EXISTS spikes (neuron INT UNSIGNED, time REAL)')

        tic = now()
        for f in glob.glob(re):
            print(f)
            while True:
                try:
                    for data in self._blockread(f):
                        self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
                        self.conn.commit()
                except:
                    continue
                break                

        toc = now()
        if self.debug: print('Inserts took %g seconds.' % (toc-tic))

        # Optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print('Indexed db in %g seconds.' % (toc-tic))