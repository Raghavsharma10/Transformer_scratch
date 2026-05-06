def version(self):
        """
        Return the version number of the Lending Club Investor tool

        Returns
        -------
        string
            The version number string
        """
        this_path = os.path.dirname(os.path.realpath(__file__))
        version_file = os.path.join(this_path, 'VERSION')
        return open(version_file).read().strip()