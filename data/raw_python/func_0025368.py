def encode_grib2_percentile(self):
        """
        Encodes member percentile data to GRIB2 format.

        Returns:
            Series of GRIB2 messages
        """
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = [1, 0, self.proj_dict['a'], 0, float(self.proj_dict['a']), 0, float(self.proj_dict['b']),
                  self.data.shape[-1], self.data.shape[-2], self.grid_dict["sw_lat"] * lscale,
                  sw_lon * lscale, 0, self.proj_dict["lat_0"] * lscale,
                  lon_0 * lscale,
                  self.grid_dict["dx"] * 1e3, self.grid_dict["dy"] * 1e3, 0b00000000, 0b01000000,
                  self.proj_dict["lat_1"] * lscale,
                  self.proj_dict["lat_2"] * lscale, -90 * lscale, 0]
        pdtmp1 = np.array([1,                # parameter category Moisture
                           31,               # parameter number Hail
                           4,                # Type of generating process Ensemble Forecast
                           0,                # Background generating process identifier
                           31,               # Generating process or model from NCEP
                           0,                # Hours after reference time data cutoff
                           0,                # Minutes after reference time data cutoff
                           1,                # Forecast time units Hours
                           0,                # Forecast time
                           1,                # Type of first fixed surface Ground
                           1,                # Scale value of first fixed surface
                           0,                # Value of first fixed surface
                           1,                # Type of second fixed surface
                           1,                # Scale value of 2nd fixed surface
                           0,                # Value of 2nd fixed surface
                           0,                # Derived forecast type
                           self.num_samples  # Number of ensemble members
                           ], dtype=np.int32)
        grib_objects = pd.Series(index=self.times, data=[None] * self.times.size, dtype=object)
        drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
        for t, time in enumerate(self.times):
            time_list = list(self.run_date.utctimetuple()[0:6])
            if grib_objects[time] is None:
                grib_objects[time] = Grib2Encode(0, np.array(grib_id_start + time_list + [2, 1], dtype=np.int32))
                grib_objects[time].addgrid(gdsinfo, gdtmp1)
            pdtmp1[8] = (time.to_pydatetime() - self.run_date).total_seconds() / 3600.0
            data = self.percentile_data[:, t] / 1000.0
            masked_data = np.ma.array(data, mask=data <= 0)
            for p, percentile in enumerate(self.percentiles):
                print("GRIB {3} Percentile {0}. Max: {1} Min: {2}".format(percentile, 
                                                                          masked_data[p].max(), 
                                                                          masked_data[p].min(),
                                                                          time))
                if percentile in range(1, 100):
                    pdtmp1[-2] = percentile
                    grib_objects[time].addfield(6, pdtmp1[:-1], 0, drtmp1, masked_data[p])
                else:
                    pdtmp1[-2] = 0
                    grib_objects[time].addfield(2, pdtmp1, 0, drtmp1, masked_data[p])
        return grib_objects