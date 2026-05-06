def set_header_info(self, r_free, r_work, resolution, title,
                        deposition_date, release_date, experimental_methods):
        """Sets the header information.
        :param r_free: the measured R-Free for the structure
        :param r_work: the measure R-Work for the structure
        :param resolution: the resolution of the structure
        :param title: the title of the structure
        :param deposition_date: the deposition date of the structure
        :param release_date: the release date of the structure
        :param experimnetal_methods: the list of experimental methods in the structure
        """
        self.r_free = r_free
        self.r_work = r_work
        self.resolution = resolution
        self.title = title
        self.deposition_date = deposition_date
        self.release_date = release_date
        self.experimental_methods = experimental_methods