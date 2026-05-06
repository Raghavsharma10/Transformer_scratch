def map(self, method: str, *args, _threaded: bool = True, **kwargs
           ) -> "AttrIndexedDict":
        "For all stored items, run a method they possess."

        work = lambda item: getattr(item, method)(*args, **kwargs)

        if _threaded:
            pool = ThreadPool(int(config.CFG["GENERAL"]["parallel_requests"]))

            try:
                pool.map(work, self.data.values())
            except KeyboardInterrupt:
                LOG.warning("CTRL-C caught, finishing current tasks...")
                pool.terminate()
            else:
                pool.close()

            pool.join()
            return self

        for item in self.data.values():
            work(item)
        return self