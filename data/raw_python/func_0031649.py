def create_from_list(self, re=[], index=True):
        """
        Create db from list of arrays.


        Parameters
        ----------
        re : list
            Index of element is cell index, and element `i` an array of spike times in ms.
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
        i = 0
        for x in re:
            data = list(zip([i] * len(x), x))
            self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
            i += 1
        self.conn.commit()
        toc = now()
        if self.debug: print('Inserts took %g seconds.' % (toc-tic))

        # Optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print('Indexed db in %g seconds.' % (toc-tic))