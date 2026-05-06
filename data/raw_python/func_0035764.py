def get_utc_timestamp(self, handle):
        """Return the UTC timestamp."""
        fpath = self._fpath_from_handle(handle)
        datetime_obj = datetime.datetime.utcfromtimestamp(
            os.stat(fpath).st_mtime
        )
        return timestamp(datetime_obj)