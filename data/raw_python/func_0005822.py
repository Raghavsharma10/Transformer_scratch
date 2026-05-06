def set_count_auto(self, count=None):
        """Sets workers count.

        By default sets it to detected number of available cores

        :param int count:
        """
        count = count or self._section.vars.CPU_CORES

        self._set('workers', count)

        return self._section