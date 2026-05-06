def write_grib2(self, path):
        """
        Writes data to grib2 file. Currently, grib codes are set by hand to hail.

        Args:
            path: Path to directory containing grib2 files.

        Returns:

        """
        if self.percentile is None:
            var_type = "mean"
        else:
            var_type = "p{0:02d}".format(self.percentile)
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = np.array([7, 1, self.proj_dict['a'], 1, self.proj_dict['a'], 1, self.proj_dict['b'],
                           self.data.shape[-2], self.data.shape[-1], self.grid_dict["sw_lat"] * lscale,
                           sw_lon * lscale, 0, self.proj_dict["lat_0"] * lscale,
                           lon_0 * lscale,
                           self.grid_dict["dx"] * 1e3, self.grid_dict["dy"] * 1e3, 0,
                           self.proj_dict["lat_1"] * lscale,
                           self.proj_dict["lat_2"] * lscale, 0, 0], dtype=np.int32)
        pdtmp1 = np.array([1, 31, 2, 0, 116, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 192, 0, self.data.shape[0]], dtype=np.int32)
        for m, member in enumerate(self.members):
            pdtmp1[-2] = m
            for t, time in enumerate(self.times):
                time_list = list(time.utctimetuple()[0:6])
                grbe = Grib2Encode(0, np.array(grib_id_start + time_list + [2, 1], dtype=np.int32))
                grbe.addgrid(gdsinfo, gdtmp1)
                pdtmp1[8] = (time.to_pydatetime() - self.run_date).total_seconds() / 3600.0
                drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
                data = self.data[m, t].astype(np.float32) / 1000.0
                masked_data = np.ma.array(data, mask=data <= 0)
                grbe.addfield(1, pdtmp1, 0, drtmp1, masked_data)
                grbe.end()
                filename = path + "{0}_{1}_mlhail_{2}_{3}.grib2".format(self.ensemble_name.replace(" ", "-"), member,
                                                                        var_type,
                                                                        time.to_datetime().strftime("%Y%m%d%H%M"))
                print("Writing to " + filename)
                grib_file = open(filename, "wb")
                grib_file.write(grbe.msg)
                grib_file.close()
        return