def neurons(self):
        """
        Return list of neuron indices.


        Parameters
        ----------
        None
        

        Returns
        -------
        list
            list of neuron indices
        
        
        See also
        --------
        sqlite3.connect.cursor
        
        """
        self.cursor.execute('SELECT DISTINCT neuron FROM spikes ORDER BY neuron')
        sel = self.cursor.fetchall()
        return np.array(sel).flatten()