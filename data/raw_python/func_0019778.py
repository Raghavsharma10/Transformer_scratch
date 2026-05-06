def getProcStatStatus(self, threads=False, **kwargs):
        """Return process counts per status and priority.
        
        @param **kwargs: Keyword variables are used for filtering the results
                         depending on the values of the columns. Each keyword 
                         must correspond to a field name with an optional 
                         suffix:
                         field:          Field equal to value or in list of 
                                         values.
                         field_ic:       Field equal to value or in list of 
                                         values, using case insensitive 
                                         comparison.
                         field_regex:    Field matches regex value or matches
                                         with any regex in list of values.
                         field_ic_regex: Field matches regex value or matches
                                         with any regex in list of values 
                                         using case insensitive match.
        @return: Dictionary of process counters.
        
        """
        procs = self.getProcList(['stat',], threads=threads, **kwargs)
        status = dict(zip(procStatusNames.values(), 
                          [0,] * len(procStatusNames)))
        prio = {'high': 0, 'low': 0, 'norm': 0, 'locked_in_mem': 0}
        total = 0
        locked_in_mem = 0
        if procs is not None:
            for cols in procs['stats']:
                col_stat = cols[0]
                status[procStatusNames[col_stat[0]]] += 1
                if '<' in col_stat[1:]:
                    prio['high'] += 1
                elif 'N' in col_stat[1:]:
                    prio['low'] += 1
                else:
                    prio['norm'] += 1
                if 'L' in col_stat[1:]:
                    locked_in_mem += 1
                total += 1
        return {'status': status, 
                'prio': prio, 
                'locked_in_mem': locked_in_mem, 
                'total': total}