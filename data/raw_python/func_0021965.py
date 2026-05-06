def analyze_log_file(logfile, pattern, reverse_paths=True, progress=True):
    "Given a log file and regex group and extract the performance data"
    if progress:
        lines = count_lines_in(logfile)
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=lines+1).start()
        counter = 0
    
    data = {}
    
    compiled_pattern = compile(pattern)
    for line in fileinput.input([logfile]):
        
        if progress:
            counter = counter + 1
        
        parsed = compiled_pattern.findall(line)[0]
        date = parsed[0]
        method = parsed[1]
        path = parsed[2]
        status = parsed[3]
        time = parsed[4]
        sql = parsed[5]
        sqltime = parsed[6]

        try:
            ignore = False
            for ignored_path in IGNORE_PATHS:
                compiled_path = compile(ignored_path)
                if compiled_path.match(path):
                    ignore = True
            if not ignore:
                if reverse_paths:
                    view = view_name_from(path)
                else:
                    view = path
                key = "%s-%s-%s" % (view, status, method)
                try:
                    data[key]['count'] = data[key]['count'] + 1
                    data[key]['times'].append(float(time))
                    data[key]['sql'].append(int(sql))
                    data[key]['sqltime'].append(float(sqltime))
                except KeyError:
                    data[key] = {
                        'count': 1,
                        'status': status,
                        'view': view,
                        'method': method,
                        'times': [float(time)],
                        'sql': [int(sql)],
                        'sqltime': [float(sqltime)],
                    }
        except Resolver404:
            pass
        
        if progress:
            pbar.update(counter)
    
    if progress:
        pbar.finish()
    
    return data