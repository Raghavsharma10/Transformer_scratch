def parseProcCmd(self, fields=('pid', 'user', 'cmd',), threads=False):
        """Execute ps command with custom output format with columns from 
        fields and return result as a nested list.
        
        The Standard Format Specifiers from ps man page must be used for the
        fields parameter.
        
        @param fields:  List of fields included in the output.
                        Default: pid, user, cmd
        @param threads: If True, include threads in output. 
        @return:        List of headers and list of rows and columns.
        
        """
        args = []
        headers = [f.lower() for f in fields]
        args.append('--no-headers')
        args.append('-e')
        if threads:
            args.append('-T')
        field_ranges = []
        fmt_strs = []
        start = 0
        for header in headers:
            field_width = psFieldWidth.get(header, psDefaultFieldWidth)
            fmt_strs.append('%s:%d' % (header, field_width))
            end = start + field_width + 1
            field_ranges.append((start,end))
            start = end
        args.append('-o')
        args.append(','.join(fmt_strs))
        lines = self.execProcCmd(*args)
        if len(lines) > 0:
            stats = []
            for line in lines:
                cols = []
                for (start, end) in field_ranges:
                    cols.append(line[start:end].strip())
                stats.append(cols)
            return {'headers': headers, 'stats': stats}
        else:
            return None