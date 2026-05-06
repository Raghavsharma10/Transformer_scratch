def summarize(group, fs=None, include_source=True):
        """
        Tabulate and write the results of ComparisonBenchmarks to a file or standard out.
        :param str group: name of the comparison group.
        :param fs: file-like object (Optional)
        """
        _line_break = '{0:-<120}\n'.format('')
        tests = sorted(ComparisonBenchmark.groups[group], key=lambda t: getattr(t, 'time_average_seconds'))
        log = StringIO.StringIO()
        log.write('Call statement:\n\n')
        log.write('\t' + tests[0].stmt)
        log.write('\n\n\n')
        fmt = "{0: <8} {1: <35} {2: <12} {3: <15} {4: <15} {5: <14}\n"
        log.write(fmt.format('Rank', 'Function Name', 'Time', '% of Slowest', 'timeit_repeat', 'timeit_number'))
        log.write(_line_break)
        log.write('\n')

        for i, t in enumerate(tests):
            func_name = "{}.{}".format(t.classname, t.callable.__name__) if t.classname else t.callable.__name__
            if i == len(tests)-1:
                time_percent = 'Slowest'
            else:
                time_percent = "{:.1f}".format(t.time_average_seconds / tests[-1].time_average_seconds * 100)
            log.write(fmt.format(i+1,
                                 func_name,
                                 convert_time_units(t.time_average_seconds),
                                 time_percent,
                                 t.timeit_repeat,
                                 t.timeit_number))
        log.write(_line_break)

        if include_source:
            log.write('\n\n\nSource Code:\n')
            log.write(_line_break)
            for test in tests:
                log.write(test.log.getvalue())
                log.write(_line_break)

        if isinstance(fs, str):
            with open(fs, 'w') as f:
                f.write(log.getvalue())

        elif fs is None:
            print(log.getvalue())
        else:
            try:
                fs.write(log.getvalue())
            except AttributeError as e:
                print(e)