def __run_blast(blast_command, input_file, *args, **kwargs):
    '''
    Run a blast variant on the given input file.
    '''

    # XXX: Eventually, translate results on the fly as requested? Or
    #      just always use our parsed object?
    if 'outfmt' in kwargs:
        raise Exception('Use of the -outfmt option is not supported')

    num_processes = kwargs.get(
        'pb_num_processes', os.sysconf('SC_NPROCESSORS_ONLN'))
    fields = kwargs.get('pb_fields', DEFAULT_HIT_FIELDS)

    blast_args = [blast_command]
    blast_args += ['-outfmt', '7 {}'.format(' '.join(fields))]
    for a in args:
        blast_args += ['-' + a]
    for k, v in kwargs.iteritems():
        if not k.startswith('pb_'):
            blast_args += ['-' + k, str(v)]

    popens = []
    for _ in range(num_processes):
        popens.append(
            subprocess.Popen(
                args=blast_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, close_fds=True))

    try:
        for r in __run_blast_select_loop(input_file, popens, fields):
            yield r
    finally:
        for p in popens:
            if p.poll() is None:
                p.terminate()
            p.wait()