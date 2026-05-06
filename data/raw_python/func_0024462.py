def runInParallel(*fns):
        """
        Runs multiple processes in parallel.

        :type: fns: def
        """
        proc = []
        for fn in fns:
            p = Process(target=fn)
            p.start()
            proc.append(p)
        for p in proc:
            p.join()