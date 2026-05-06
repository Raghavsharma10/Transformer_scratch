def reporting(self):
        """
        report on consumption info
        """
        self.thread_debug("reporting")
        res = resource.getrusage(resource.RUSAGE_SELF)
        self.NOTIFY("",
                    type='internal-usage',
                    maxrss=round(res.ru_maxrss/1024, 2),
                    ixrss=round(res.ru_ixrss/1024, 2),
                    idrss=round(res.ru_idrss/1024, 2),
                    isrss=round(res.ru_isrss/1024, 2),
                    threads=threading.active_count(),
                    proctot=len(self.monitors),
                    procwin=self.stats.procwin)