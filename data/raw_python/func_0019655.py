def registerFilters(self, **kwargs):
        """Register multiple filters at once.
        
        @param **kwargs: Multiple filters are registered using keyword 
                         variables. Each keyword must correspond to a field name 
                         with an optional suffix:
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
        """
        for (key, patterns) in kwargs.items():
            if key.endswith('_regex'):
                col = key[:-len('_regex')]
                is_regex = True
            else:
                col = key
                is_regex = False
            if col.endswith('_ic'):
                col = col[:-len('_ic')]
                ignore_case = True
            else:
                ignore_case = False
            self.registerFilter(col, patterns, is_regex, ignore_case)