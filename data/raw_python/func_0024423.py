def _scan_fpatterns(self, state):
        '''
        For a list of given fpatterns, this starts a thread
        collecting log lines from file

        >>> os.path.isfile = lambda path: path == '/path/to/log_file.log'
        >>> lc = LogCollector('file=/path/to/log_file.log:formatter=logagg.formatters.basescript', 30)

        >>> print(lc.fpaths)
        file=/path/to/log_file.log:formatter=logagg.formatters.basescript

        >>> print('formatters loaded:', lc.formatters)
        {}
        >>> print('log file reader threads started:', lc.log_reader_threads)
        {}
        >>> state = AttrDict(files_tracked=list())
        >>> print('files bieng tracked:', state.files_tracked)
        []


        >>> if not state.files_tracked:
        >>>     lc._scan_fpatterns(state)
        >>>     print('formatters loaded:', lc.formatters)
        >>>     print('log file reader threads started:', lc.log_reader_threads)
        >>>     print('files bieng tracked:', state.files_tracked)


        '''
        for f in self.fpaths:
            fpattern, formatter =(a.split('=')[1] for a in f.split(':', 1))
            self.log.debug('scan_fpatterns', fpattern=fpattern, formatter=formatter)
            # TODO code for scanning fpatterns for the files not yet present goes here
            fpaths = glob.glob(fpattern)
            # Load formatter_fn if not in list
            fpaths = list(set(fpaths) - set(state.files_tracked))
            for fpath in fpaths:
                try:
                    formatter_fn = self.formatters.get(formatter,
                                  load_formatter_fn(formatter))
                    self.log.info('found_formatter_fn', fn=formatter)
                    self.formatters[formatter] = formatter_fn
                except (SystemExit, KeyboardInterrupt): raise
                except (ImportError, AttributeError):
                    self.log.exception('formatter_fn_not_found', fn=formatter)
                    sys.exit(-1)
                # Start a thread for every file
                self.log.info('found_log_file', log_file=fpath)
                log_f = dict(fpath=fpath, fpattern=fpattern,
                                formatter=formatter, formatter_fn=formatter_fn)
                log_key = (fpath, fpattern, formatter)
                if log_key not in self.log_reader_threads:
                    self.log.info('starting_collect_log_lines_thread', log_key=log_key)
                    # There is no existing thread tracking this log file. Start one
                    log_reader_thread = util.start_daemon_thread(self.collect_log_lines, (log_f,))
                    self.log_reader_threads[log_key] = log_reader_thread
                state.files_tracked.append(fpath)
        time.sleep(self.SCAN_FPATTERNS_INTERVAL)