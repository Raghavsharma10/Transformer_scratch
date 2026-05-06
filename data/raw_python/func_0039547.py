def return_labels_numpy(self, original=False):
        """
        Returns a 2d numpy array of labels

        Parameters
        ----------
        original : if True, will return original labels, if False, will return transformed labels (as defined by
        label_dict), default value: False

        Returns
        -------
        A numpy array of labels, each row corresponds to a single datapoint
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            engine = create_engine('sqlite:////' + self.dbpath)
            trainset.Base.metadata.create_all(engine)
            session_cl = sessionmaker(bind=engine)
            session = session_cl()
            tmp_object = session.query(trainset.TrainSet).get(1)

            columns_amt = len(tmp_object.labels['original'])
            return_array = np.zeros([self.points_amt, columns_amt])
            for i in enumerate(session.query(trainset.TrainSet).order_by(trainset.TrainSet.id)):
                if original is False:
                    return_array[i[0], :] = i[1].labels['transformed']
                else:
                    return_array[i[0], :] = i[1].labels['original']
            session.close()
            return return_array