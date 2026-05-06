def get_battery_state(self, prop):
        """
        Return the first line from the file located at battery_path/prop as a
        string.
        """
        with open(os.path.join(self.options['battery_path'], prop), 'r') as f:
                return f.readline().strip()