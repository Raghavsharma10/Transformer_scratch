async def exec_loop(stdout, stderr, kernel, mode, code, *, opts=None,
                    vprint_done=print_done, is_multi=False):
    '''
    Fully streamed asynchronous version of the execute loop.
    '''
    async with kernel.stream_execute(code, mode=mode, opts=opts) as stream:
        async for result in stream:
            if result.type == aiohttp.WSMsgType.TEXT:
                result = json.loads(result.data)
            else:
                # future extension
                continue
            for rec in result.get('console', []):
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
                msg = 'Clean finished. (exit code = {0})'.format(exitCode)
                if is_multi:
                    print(msg, file=stderr)
                vprint_done(msg)
            elif result['status'] == 'build-finished':
                exitCode = result.get('exitCode')
                msg = 'Build finished. (exit code = {0})'.format(exitCode)
                if is_multi:
                    print(msg, file=stderr)
                vprint_done(msg)
            elif result['status'] == 'finished':
                exitCode = result.get('exitCode')
                msg = 'Execution finished. (exit code = {0})'.format(exitCode)
                if is_multi:
                    print(msg, file=stderr)
                vprint_done(msg)
                break
            elif result['status'] == 'waiting-input':
                if result['options'].get('is_password', False):
                    code = getpass.getpass()
                else:
                    code = input()
                await stream.send_str(code)
            elif result['status'] == 'continued':
                pass