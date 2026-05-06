def run_cmake(command, build_path, default_build_path):
    """
    Execute CMake command.
    """
    from subprocess import Popen, PIPE
    from shutil import rmtree

    topdir = os.getcwd()
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout_coded, stderr_coded = p.communicate()
    stdout = stdout_coded.decode('UTF-8')
    stderr = stderr_coded.decode('UTF-8')

    # print cmake output to screen
    print(stdout)

    if stderr:
        # we write out stderr but we do not stop yet
        # this is because CMake warnings are sent to stderr
        # and they might be benign
        sys.stderr.write(stderr)

    # write cmake output to file
    with open(os.path.join(build_path, 'cmake_output'), 'w') as f:
        f.write(stdout)

    # change directory and return
    os.chdir(topdir)

    # to figure out whether configuration was a success
    # we check for 3 sentences that should be part of stdout
    configuring_done = '-- Configuring done' in stdout
    generating_done = '-- Generating done' in stdout
    build_files_written = '-- Build files have been written to' in stdout
    configuration_successful = configuring_done and generating_done and build_files_written

    if configuration_successful:
        save_setup_command(sys.argv, build_path)
        print_build_help(build_path, default_build_path)
    else:
        if (build_path == default_build_path):
            # remove build_path iff not set by the user
            # otherwise removal can be dangerous
            rmtree(default_build_path)