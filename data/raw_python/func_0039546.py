def return_labels(self, original=False):
        """
        Returns the labels of the dataset

        Parameters
        ----------
        original : if True, will return original labels, if False, will return transformed labels (as defined by
        label_dict), default value: False

        Returns
        -------
        A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
        label
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            engine = create_engine('sqlite:////' + self.dbpath)
            trainset.Base.metadata.create_all(engine)
            session_cl = sessionmaker(bind=engine)
            session = session_cl()
            return_list = []
            for i in session.query(trainset.TrainSet).order_by(trainset.TrainSet.id):
                if original is True:
                    row_list = i.labels['original']
                else:
                    row_list = i.labels['transformed']
                return_list.append(row_list[:])
            session.close()
            return return_list