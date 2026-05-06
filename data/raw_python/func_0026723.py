def _file_nums_str(self, n_all, n_type, n_ign):
        """Construct a string showing the number of different file types.

        Returns
        -------
        f_str : str
        """
        # 'other' is the difference between all and named
        n_oth = n_all - np.sum(n_type)

        f_str = "{} Files".format(n_all) + " ("
        if len(n_type):
            f_str += ", ".join("{} {}".format(name, num) for name, num in
                               zip(self._COUNT_FILE_TYPES, n_type))
            f_str += ", "
        f_str += "other {}; {} ignored)".format(n_oth, n_ign)
        return f_str