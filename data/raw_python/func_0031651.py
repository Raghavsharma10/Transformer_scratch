def interval(self, T=[0, 1000]):
        """
        Get all spikes in a time interval T.


        Parameters
        ----------
        T : list
            Time interval.


        Returns
        -------
        s : list
            Nested list with spike times.



        See also
        --------
        sqlite3.connect.cursor
        
        """
        self.cursor.execute('SELECT * FROM spikes WHERE time BETWEEN %f AND %f' % tuple(T))
        sel = self.cursor.fetchall()
        return sel