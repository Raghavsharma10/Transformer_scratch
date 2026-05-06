def load_data(self):
        """
        Loads data from each ensemble member.
        """
        for m, member in enumerate(self.members):
            mo = ModelOutput(self.ensemble_name, member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.map_file, self.single_step)
            mo.load_data()
            if self.data is None:
                self.data = np.zeros((len(self.members), mo.data.shape[0], mo.data.shape[1], mo.data.shape[2]),
                                     dtype=np.float32)
            if mo.units == "m":
                self.data[m] = mo.data * 1000
                self.units = "mm"
            else:
                self.data[m] = mo.data
            if self.units == "":
                self.units = mo.units
            del mo.data
            del mo