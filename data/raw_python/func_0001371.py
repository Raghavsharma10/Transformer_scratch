def exec_all_endpoints(self, *args, **kwargs):
        """Execute each passed endpoint and collect the results. If a result
        is anoter `MultipleResults` it will extend the results with those
        contained therein. If the result is `NoResult`, skip the addition."""
        results = []
        for handler in self.endpoints:
            if isinstance(handler, weakref.ref):
                handler = handler()
            if self.adapt_params:
                bind = self._adapt_call_params(handler, args, kwargs)
                res = handler(*bind.args, **bind.kwargs)
            else:
                res = handler(*args, **kwargs)
            if isinstance(res, MultipleResults):
                if res.done:
                    results += res.results
                else:
                    results += res._results
            elif res is not NoResult:
                results.append(res)
        return MultipleResults(results, concurrent=self.concurrent, owner=self)