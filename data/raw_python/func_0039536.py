def prepopulate(self):
        """
        Creates a database file (if it doesn't exist, writes each data point's path, real_id into it)

        Parameters
        ----------
        self

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            engine = create_engine('sqlite:////' + self.dbpath)
            self._db_base.metadata.create_all(engine)
            self._prepopulated = True
            session_cl = sessionmaker(bind=engine)
            session = session_cl()

            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                for f_name in filenames:
                    datapoint = self._set_object(real_id=cutoff_filename(self.file_prefix, self.file_suffix, f_name),
                                                 path=f_name, features=None)
                    session.add(datapoint)
                    self.points_amt += 1
            session.commit()
            session.close()
        return None