def exec_loop_sync(stdout, stderr, kernel, mode, code, *, opts=None,
                   vprint_done=print_done):
    '''
    Old synchronous polling version of the execute loop.
    '''
    opts = opts if opts else {}
    run_id = None  # use server-assigned run ID
    while True:
        result = kernel.execute(run_id, code, mode=mode, opts=opts)
        run_id = result['runId']
        opts.clear()  # used only once
        for rec in result['console']:
            if rec[0] == 'stdout':
                print(rec[1], end='', file=stdout)
            elif rec[0] == 'stderr':
                print(rec[1], end='', file=stderr)
            else:
                print('----- output record (type: {0}) -----'.format(rec[0]),
                      file=stdout)
                print(rec[1], file=stdout)
                print('----- end of record -----', file=stdout)
        stdout.flush()
        files = result.get('files', [])
        if files:
            print('--- generated files ---', file=stdout)
            for item in files:
                print('{0}: {1}'.format(item['name'], item['url']), file=stdout)
            print('--- end of generated files ---', file=stdout)
        if result['status'] == 'clean-finished':
            exitCode = result.get('exitCode')
            vprint_done('Clean finished. (exit code = {0}'.format(exitCode),
                        file=stdout)
            mode = 'continue'
            code = ''
        elif result['status'] == 'build-finished':
            exitCode = result.get('exitCode')
            vprint_done('Build finished. (exit code = {0})'.format(exitCode),
                        file=stdout)
            mode = 'continue'
            code = ''
        elif result['status'] == 'finished':
            exitCode = result.get('exitCode')
            vprint_done('Execution finished. (exit code = {0})'.format(exitCode),
                        file=stdout)
            break
        elif result['status'] == 'waiting-input':
            mode = 'input'
            if result['options'].get('is_password', False):
                code = getpass.getpass()
            else:
                code = input()
        elif result['status'] == 'continued':
            mode = 'continue'
            code = ''