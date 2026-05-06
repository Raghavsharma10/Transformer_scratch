def registerFilter(self, column, patterns, is_regex=False, 
                       ignore_case=False):
        """Register filter on a column of table.
        
        @param column:      The column name.
        @param patterns:    A single pattern or a list of patterns used for 
                            matching column values.
        @param is_regex:    The patterns will be treated as regex if True, the 
                            column values will be tested for equality with the
                            patterns otherwise.
        @param ignore_case: Case insensitive matching will be used if True.
        
        """
        if isinstance(patterns, basestring):
            patt_list = (patterns,)
        elif isinstance(patterns, (tuple, list)):
            patt_list = list(patterns)
        else:
            raise ValueError("The patterns parameter must either be as string "
                             "or a tuple / list of strings.")
        if is_regex:
            if ignore_case:
                flags = re.IGNORECASE
            else:
                flags = 0
            patt_exprs = [re.compile(pattern, flags) for pattern in patt_list]
        else:
            if ignore_case:
                patt_exprs = [pattern.lower() for pattern in patt_list]
            else:
                patt_exprs = patt_list
        self._filters[column] = (patt_exprs, is_regex, ignore_case)