def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)