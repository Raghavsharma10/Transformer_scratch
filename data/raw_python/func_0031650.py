def select(self, neurons):
        """
        Select spike trains.


        Parameters
        ----------
        neurons : numpy.ndarray or list
            Array of list of neurons.


        Returns
        -------
        list
            List of numpy.ndarray objects containing spike times.


        See also
        --------
        sqlite3.connect.cursor
        
        """
        s = []
        for neuron in neurons:
            self.cursor.execute('SELECT time FROM spikes where neuron = %d' % neuron)
            sel = self.cursor.fetchall()
            spikes = np.array(sel).flatten()
            s.append(spikes)
        return s