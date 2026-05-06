def _get_minutes_from_last_update(self, time):
        """ How much minutes passed from last update to given time """
        time_from_last_update = time - self.last_update_time
        return int(time_from_last_update.total_seconds() / 60)