def return_single_labels(self, object_id):
        """
        Returns all labels for an object specified by the object_id

        Parameters
        ----------
        object_id : int, id of object in database

        Returns
        -------
        result : list of labels
        """
        engine = create_engine('sqlite:////' + self.dbpath)
        trainset.Base.metadata.create_all(engine)
        session_cl = sessionmaker(bind=engine)
        session = session_cl()
        tmp_object = session.query(trainset.TrainSet).get(object_id)
        return tmp_object.labels