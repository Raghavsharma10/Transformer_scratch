def to_csv(self, file):
        """
        Write all the trajectories of a collection to a csv file with the headers 'description', 'time' and 'value'.

        :param file: a file object to write to
        :type file: :class:`file`
        :return:
        """
        file.write("description,time,value\n")
        for traj in self:
            for t,v in traj:
                file.write("%s,%f,%f\n"% (traj.description.symbol, t, v))