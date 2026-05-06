def verify_param(self, param, must=[], r=None):
        '''return Code.ARGUMENT_MISSING if every key in must not found in param'''
        if APIKEY not in param:
            param[APIKEY] = self.apikey()

        r = Result() if r is None else r
        for p in must:
            if p not in param:
                r.code(Code.ARGUMENT_MISSING).detail('missing-' + p)
                break

        return r