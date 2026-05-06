def get_high_water_mark(self, mark_type, obstory_name=None):
        """
        Retrieves the high water mark for a given obstory, defaulting to the current installation ID

        :param string mark_type:
            The type of high water mark to set
        :param string obstory_name:
            The obstory ID to check for, or the default installation ID if not specified
        :return:
            A UTC datetime for the high water mark, or None if none was found.
        """
        if obstory_name is None:
            obstory_name = self.obstory_name

        obstory = self.get_obstory_from_name(obstory_name)
        key_id = self.get_hwm_key_id(mark_type)

        self.con.execute('SELECT time FROM archive_highWaterMarks WHERE markType=%s AND observatoryId=%s',
                         (key_id, obstory['uid']))
        results = self.con.fetchall()
        if len(results) > 0:
            return results[0]['time']
        return None