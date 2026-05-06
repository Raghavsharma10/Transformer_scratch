def to_csv(self, file):
        """
        Write this trajectory to a csv file with the headers 'time' and 'value'.

        :param file: a file object to write to
        :type file: :class:`file`
        :return:
        """

        file.write("time,value\n")
        for t,v in self:
            file.write("%f,%f\n"% (t, v))