def add_file_argument(self, *args, **kwargs):
        """ Add an argument that represents the location of a file

        :param args:
        :param kwargs:
        :return:
        """
        rval = self.add_argument(*args, **kwargs)
        self.file_args.append(rval)
        return rval