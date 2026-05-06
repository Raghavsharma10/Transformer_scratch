def getProcDict(self, fields=('user', 'cmd',), threads=False, **kwargs):
        """Execute ps command with custom output format with columns format with 
        columns from fields, and return result as a nested dictionary with 
        the key PID or SPID.
        
        The Standard Format Specifiers from ps man page must be used for the 
        fields parameter.
        
        @param fields:   Fields included in the output.
                         Default: user, cmd
                         (PID or SPID column is included by default.)
        @param threads:  If True, include threads in output.
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
        @return:         Nested dictionary indexed by:
                           PID for process info.
                           SPID for thread info.
        
        """
        stats = {}
        field_list = list(fields)
        num_cols = len(field_list)
        if threads:
            key = 'spid'
        else:
            key = 'pid'
        try:
            key_idx = field_list.index(key)
        except ValueError:
            field_list.append(key)
            key_idx = len(field_list) - 1
        result = self.getProcList(field_list, threads, **kwargs)
        if result is not None:
            headers = result['headers'][:num_cols]
            lines = result['stats']
            if len(lines) > 1:
                for cols in lines:
                    stats[cols[key_idx]] = dict(zip(headers, cols[:num_cols]))
            return stats
        else:
            return None