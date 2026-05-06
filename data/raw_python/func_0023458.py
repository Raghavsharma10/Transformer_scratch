def write(self, log, rotate=False):
        """Write the output of all finished processes to a compiled log file."""
        # Get path for logfile
        if rotate:
            timestamp = time.strftime('-%Y%m%d-%H%M')
            logPath = os.path.join(self.log_dir, 'queue{}.log'.format(timestamp))
        else:
            logPath = os.path.join(self.log_dir, 'queue.log')

        # Remove existing Log
        if os.path.exists(logPath):
            os.remove(logPath)

        log_file = open(logPath, 'w')
        log_file.write('Pueue log for executed Commands: \n \n')

        # Format, color and write log
        for key, logentry in log.items():
            if logentry.get('returncode') is not None:
                try:
                    # Get returncode color:
                    returncode = logentry['returncode']
                    if returncode == 0:
                        returncode = Color('{autogreen}' + '{}'.format(returncode) + '{/autogreen}')
                    else:
                        returncode = Color('{autored}' + '{}'.format(returncode) + '{/autored}')

                    # Write command id with returncode and actual command
                    log_file.write(
                        Color('{autoyellow}' + 'Command #{} '.format(key) + '{/autoyellow}') +
                        'exited with returncode {}: \n'.format(returncode) +
                        '"{}" \n'.format(logentry['command'])
                    )
                    # Write path
                    log_file.write('Path: {} \n'.format(logentry['path']))
                    # Write times
                    log_file.write('Start: {}, End: {} \n'
                                   .format(logentry['start'], logentry['end']))

                    # Write STDERR
                    if logentry['stderr']:
                        log_file.write(Color('{autored}Stderr output: {/autored}\n    ') + logentry['stderr'])

                    # Write STDOUT
                    if len(logentry['stdout']) > 0:
                        log_file.write(Color('{autogreen}Stdout output: {/autogreen}\n    ') + logentry['stdout'])

                    log_file.write('\n')
                except Exception as a:
                    print('Failed while writing to log file. Wrong file permissions?')
                    print('Exception: {}'.format(str(a)))

        log_file.close()