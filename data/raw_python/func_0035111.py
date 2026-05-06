def txt_read_in(self):
        """Read in txt files.

        Method for reading in text or csv files. This uses ascii class from astropy.io
        for flexible input. It is slower than numpy, but has greater flexibility with less input.

        """

        # read in
        data = ascii.read(self.WORKING_DIRECTORY + '/' + self.file_name)

        # find number of distinct x and y points.
        num_x_pts = len(np.unique(data[self.x_column_label]))
        num_y_pts = len(np.unique(data[self.y_column_label]))

        # create 2D arrays of x,y,z
        self.xvals = np.reshape(np.asarray(data[self.x_column_label]), (num_y_pts, num_x_pts))
        self.yvals = np.reshape(np.asarray(data[self.y_column_label]), (num_y_pts, num_x_pts))
        self.zvals = np.reshape(np.asarray(data[self.z_column_label]), (num_y_pts, num_x_pts))

        return