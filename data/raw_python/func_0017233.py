def rolling(self, op):
        """Fast rolling operation with O(log n) updates where n is the
        window size
        """
        missing   = self.missing
        ismissing = self.ismissing
        window = self.window
        it = iter(self.iterable)
        queue = deque(islice(it, window))
        ol = self.skiplist((e for e in queue if e == e))
        yield op(ol,missing)
        for newelem in it:
            oldelem = queue.popleft()
            if not ismissing(oldelem):
                ol.remove(oldelem)
            queue.append(newelem)
            if not ismissing(newelem):
                ol.insert(newelem)
            yield op(ol, missing)