def num_spikes(self):
        """
        Return total number of spikes.


        Parameters
        ----------
        None
        
        
        Returns
        -------
        list

        """
        self.cursor.execute('SELECT Count(*) from spikes')
        rows = self.cursor.fetchall()[0]
        # Check against 'wc -l *ex*.gdf'
        if self.debug: print('DB has %d spikes' % rows)
        return rows