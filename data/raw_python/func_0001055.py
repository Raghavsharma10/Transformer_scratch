def run_mecab_process(content, *args, **kwargs):
    ''' Use subprocess to run mecab '''
    encoding = 'utf-8' if 'encoding' not in kwargs else kwargs['encoding']
    mecab_loc = kwargs['mecab_loc'] if 'mecab_loc' in kwargs else None
    if mecab_loc is None:
        mecab_loc = MECAB_LOC
    proc_args = [mecab_loc]
    if args:
        proc_args.extend(args)
    output = subprocess.run(proc_args,
                            input=content.encode(encoding),
                            stdout=subprocess.PIPE)
    output_string = os.linesep.join(output.stdout.decode(encoding).splitlines())
    return output_string