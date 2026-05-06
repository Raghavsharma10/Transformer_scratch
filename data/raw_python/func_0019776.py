def getProcList(self, fields=('pid', 'user', 'cmd',), threads=False,
                    **kwargs):
        """Execute ps command with custom output format with columns columns 
        from fields, select lines using the filters defined by kwargs and return 
        result as a nested list.
        
        The Standard Format Specifiers from ps man page must be used for the
        fields parameter.
        
        @param fields:   Fields included in the output.
                         Default: pid, user, cmd
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
        @return:         List of headers and list of rows and columns.
        
        """
        field_list = list(fields)
        for key in kwargs:
            col = re.sub('(_ic)?(_regex)?$', '', key)
            if not col in field_list:
                field_list.append(col)
        pinfo = self.parseProcCmd(field_list, threads)
        if pinfo:
            if len(kwargs) > 0:
                pfilter = util.TableFilter()
                pfilter.registerFilters(**kwargs)
                stats = pfilter.applyFilters(pinfo['headers'], pinfo['stats'])
                return {'headers': pinfo['headers'], 'stats': stats}
            else:
                return pinfo
        else:
            return None