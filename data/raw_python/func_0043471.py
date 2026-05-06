def which(program):
    " Check program is exists. "

    head, _ = op.split(program)

    if head:
        if is_exe(program):
            return program
    else:
        for path in environ["PATH"].split(pathsep):
            exe_file = op.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None