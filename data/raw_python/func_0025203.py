def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        trace_map = {}
        self._trace(n, trace_map)
        s = list(trace_map.keys())
        s.sort()
        return s