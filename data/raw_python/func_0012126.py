def partial_results(self):
        '''The results that the RPC has received *so far*

        This may also be the complete results if :attr:`complete` is ``True``.
        '''
        results = []
        for r in self._results:
            if isinstance(r, Exception):
                results.append(type(r)(*deepcopy(r.args)))
            elif hasattr(r, "__iter__") and not hasattr(r, "__len__"):
                # pass generators straight through
                results.append(r)
            else:
                results.append(deepcopy(r))
        return results