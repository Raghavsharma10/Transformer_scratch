def map(self, fn, *seq):
        "Perform a map operation distributed among the workers. Will "
        "block until done."
        results = Queue()
        args = zip(*seq)
        for seq in args:
            j = SimpleJob(results, fn, seq)
            self.put(j)

        # Aggregate results
        r = []
        for i in range(len(list(args))):
            r.append(results.get())

        return r