def update_data(self):
        """This is a method that will be called every time a packet is opened
        from the roaster."""
        time_elapsed = datetime.datetime.now() - self.start_time
        crntTemp = self.roaster.current_temp
        targetTemp = self.roaster.target_temp
        heaterLevel = self.roaster.heater_level
        # print(
        #     "Time: %4.6f, crntTemp: %d, targetTemp: %d, heaterLevel: %d" %
        #     (time_elapsed.total_seconds(), crntTemp, targetTemp, heaterLevel))
        self.file.write(
            "%4.6f,%d,%d,%d\n" %
            (time_elapsed.total_seconds(), crntTemp, targetTemp, heaterLevel))