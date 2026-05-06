def hdf5_read_in(self):
        """Method for reading in hdf5 files.

        """

        with h5py.File(self.WORKING_DIRECTORY + '/' + self.file_name) as f:

            # read in
            data = f['data']

            # find number of distinct x and y points.
            num_x_pts = len(np.unique(data[self.x_column_label][:]))
            num_y_pts = len(np.unique(data[self.y_column_label][:]))

            # create 2D arrays of x,y,z
            self.xvals = np.reshape(data[self.x_column_label][:], (num_y_pts, num_x_pts))
            self.yvals = np.reshape(data[self.y_column_label][:], (num_y_pts, num_x_pts))
            self.zvals = np.reshape(data[self.z_column_label][:], (num_y_pts, num_x_pts))
        return