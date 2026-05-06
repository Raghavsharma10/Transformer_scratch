def _bracket(self, qinit, f0, fun):
        """Find a bracket that does contain the minimum"""
        self.num_bracket = 0
        qa = qinit
        fa = fun(qa)
        counter = 0
        if fa >= f0:
            while True:
                self.num_bracket += 1
                #print "    bracket shrink"
                qb, fb = qa, fa
                qa /= 1+phi
                fa = fun(qa)
                if qa < self.qtol:
                    return
                if fa < f0:
                    return (0, f0), (qa, fa), (qb, fb)
                counter += 1
                if self.max_iter is not None and counter > self.max_iter:
                    return
        else:
            self.num_bracket += 1
            #print "    bracket grow1"
            qb, fb = qa, fa
            qa *= (1+phi)
            fa = fun(qa)
            if fa >= fb:
                return (0, f0), (qb, fb), (qa, fa)
            while True:
                self.num_bracket += 1
                #print "    bracket grow2"
                qc, fc = qb, fb
                qb, fb = qa, fa
                qa = qb*(1+phi) - qc
                fa = fun(qa)
                if fa >= fb:
                    return (qc, fc), (qb, fb), (qa, fa)
                counter += 1
                if self.max_iter is not None and counter > self.max_iter:
                    return