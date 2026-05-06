def load_data(self):
        """
        Loads data from MRMS GRIB2 files and handles compression duties if files are compressed.
        """
        data = []
        loaded_dates = []
        loaded_indices = []
        for t, timestamp in enumerate(self.all_dates):
            date_str = timestamp.date().strftime("%Y%m%d")
            full_path = self.path_start + date_str + "/"
            if self.variable in os.listdir(full_path):
                full_path += self.variable + "/"
                data_files = sorted(os.listdir(full_path))
                file_dates = pd.to_datetime([d.split("_")[-1][0:13] for d in data_files])
                if timestamp in file_dates:
                    data_file = data_files[np.where(timestamp==file_dates)[0][0]]
                    print(full_path + data_file)
                    if data_file[-2:] == "gz":
                        subprocess.call(["gunzip", full_path + data_file])
                        file_obj = Nio.open_file(full_path + data_file[:-3])
                    else:
                        file_obj = Nio.open_file(full_path + data_file)
                    var_name = sorted(file_obj.variables.keys())[0]
                    data.append(file_obj.variables[var_name][:])
                    if self.lon is None:
                        self.lon = file_obj.variables["lon_0"][:]
                        # Translates longitude values from 0:360 to -180:180
                        if np.count_nonzero(self.lon > 180) > 0:
                            self.lon -= 360
                        self.lat = file_obj.variables["lat_0"][:]
                    file_obj.close()
                    if data_file[-2:] == "gz":
                        subprocess.call(["gzip", full_path + data_file[:-3]])
                    else:
                        subprocess.call(["gzip", full_path + data_file])
                    loaded_dates.append(timestamp)
                    loaded_indices.append(t)
        if len(loaded_dates) > 0:
            self.loaded_dates = pd.DatetimeIndex(loaded_dates)
            self.data = np.ones((self.all_dates.shape[0], data[0].shape[0], data[0].shape[1])) * -9999
            self.data[loaded_indices] = np.array(data)