def set_memory_limits(self, rss_soft=None, rss_hard=None):
        """Sets worker memory limits for cheapening.

        :param int rss_soft: Don't spawn new workers if total resident memory usage
            of all workers is higher than this limit in bytes.
            
            .. warning:: This option expects memory reporting enabled:
                ``.logging.set_basic_params(memory_report=1)``

        :param int rss_hard: Try to stop workers if total workers resident memory usage
            is higher that thi limit in bytes.

        """
        self._set('cheaper-rss-limit-soft', rss_soft)           
        self._set('cheaper-rss-limit-hard', rss_hard)

        return self._section