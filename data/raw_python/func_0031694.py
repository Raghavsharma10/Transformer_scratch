def collect_gdf(self):
        """
        Collect the gdf-files from network sim in folder `spike_output_path`
        into sqlite database, using the GDF-class.
        
        
        Parameters
        ----------
        None
        
        
        Returns
        -------
        None
        
        """
        # Resync
        COMM.Barrier()

        # Raise Exception if there are no gdf files to be read
        if len(glob.glob(os.path.join(self.spike_output_path,
                                      self.label + '*.'+ self.ext))) == 0:
            raise Exception('path to files contain no gdf-files!')

        #create in-memory databases of spikes
        if not hasattr(self, 'dbs'):
            self.dbs = {}
        
        for X in self.X:
            db = GDF(os.path.join(self.dbname),
                     debug=True, new_db=True)
            db.create(re=os.path.join(self.spike_output_path,
                                      '{0}*{1}*{2}'.format(self.label, X,
                                                           self.ext)),
                      index=True)
            self.dbs.update({
                    X : db
                })
      
        COMM.Barrier()