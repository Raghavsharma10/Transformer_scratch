def _parse(self, pattern):
        """Parse string of comma-separated x-y/step -like ranges"""
        # Comma separated ranges
        if pattern.find(',') < 0:
            subranges = [pattern]
        else:
            subranges = pattern.split(',')

        for subrange in subranges:
            if subrange.find('/') < 0:
                step = 1
                baserange = subrange
            else:
                baserange, step = subrange.split('/', 1)

            try:
                step = int(step)
            except ValueError:
                raise RangeSetParseError(subrange,
                        "cannot convert string to integer")

            if baserange.find('-') < 0:
                if step != 1:
                    raise RangeSetParseError(subrange,
                            "invalid step usage")
                begin = end = baserange
            else:
                begin, end = baserange.split('-', 1)

            # compute padding and return node range info tuple
            try:
                pad = 0
                if int(begin) != 0:
                    begins = begin.lstrip("0")
                    if len(begin) - len(begins) > 0:
                        pad = len(begin)
                    start = int(begins)
                else:
                    if len(begin) > 1:
                        pad = len(begin)
                    start = 0
                if int(end) != 0:
                    ends = end.lstrip("0")
                else:
                    ends = end
                stop = int(ends)
            except ValueError:
                raise RangeSetParseError(subrange,
                        "cannot convert string to integer")

            # check preconditions
            if stop > 1e100 or start > stop or step < 1:
                raise RangeSetParseError(subrange,
                                         "invalid values in range")

            self.add_range(start, stop + 1, step, pad)