def _load_stats(self):
        """ Load the webpack-stats file """
        for attempt in range(0, 3):
            try:
                with self.stats_file.open() as f:
                    return json.load(f)
            except ValueError:
                # If we failed to parse the JSON, it's possible that the
                # webpack process is writing to it concurrently and it's in a
                # bad state. Sleep and retry.
                if attempt < 2:
                    time.sleep(attempt * 0.2)
                else:
                    raise
            except IOError:
                raise IOError(
                    "Could not read stats file {0}. Make sure you are using the "
                    "webpack-bundle-tracker plugin" .format(self.stats_file))