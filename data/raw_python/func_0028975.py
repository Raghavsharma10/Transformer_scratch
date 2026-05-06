def _update(self, datapoints):
        """
        This method store in the datapoints in the current database.

            :datapoints: is a list of tupple with the epoch timestamp and value
                 [(1368977629,10)]
        """
        if len(datapoints) == 1:
            timestamp, value = datapoints[0]
            whisper.update(self.path, value, timestamp)
        else:
            whisper.update_many(self.path, datapoints)