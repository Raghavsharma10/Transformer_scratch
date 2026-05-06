def worker(data, json_file):
    """
    Handle parameter substitution and execute command as child process.
    """
    # PERHAPS TODO: Support either full or relative paths.
    with open(json_file) as fp:
        d = json.load(fp)
    json_directory = os.path.dirname(json_file)
    def p(*parts):
        return os.path.join(json_directory, *parts)

    # STDOUT and STDERR will be written to a log file in each job directory.
    log_file = data['log_file']

    # A template file will be written in each job directory, including the
    # substitution that was performed..
    savecmd_file = data['savecmd_file']

    # if a template file is being used, then we write out to it
    template_file = data['template_file']
    if template_file:
        output_template = p(os.path.basename(template_file))
        with open(output_template, 'w') as out_fobj:
            template_subs_file(template_file, out_fobj, d)

        # Copy permissions to destination
        try:
            shutil.copymode(template_file, output_template)
        except OSError as e:
            if e.errno == errno.EPERM:
                logging.warn("Couldn't set permissions on %s. "
                        "Continuing with existing permissions",
                        output_template)
            else:
                raise

    work = data['template'].format(**d)

    if savecmd_file:
        with open(p(savecmd_file), 'w') as command_file:
            command_file.write(work + "\n")

    # View what actions will take place in dry_run mode.
    if data['dry_run']:
        logging.info("%s - Dry run of %s\n", p(), work)
    else:
        try:
            with open(p(log_file), 'w') as log:
                cmd = shlex.split(work)
                while True:
                    try:
                        pr = subprocess.Popen(
                            cmd, stdout=log, stderr=log, cwd=p())
                    except OSError as e:
                        if e.errno != errno.EINTR:
                            raise
                        continue
                    else:
                        break
                logging.info('[%d] Started %s in %s', pr.pid, work, p())
                nestproc = NestlyProcess(cmd, p(), pr)
                yield nestproc
        except Exception as e:
            # Seems useful to print the command that failed to make the
            # traceback more meaningful.  Note that error output could get
            # mixed up if two processes encounter errors at the same instant
            logging.error("%s - Error executing %s - %s", p(), work, e)
            raise e