def save(self, file=CONFIG_FILE):
        """
        Save configuration to provided path as a yaml file
        """
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            yaml.dump(self._settings, f, Dumper=yaml.RoundTripDumper, width=float("inf"))