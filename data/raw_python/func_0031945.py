def sql_program_name_func(command):
    """
    Extract program name from `command`.

    >>> sql_program_name_func('ls')
    'ls'
    >>> sql_program_name_func('git status')
    'git'
    >>> sql_program_name_func('EMACS=emacs make')
    'make'

    :type command: str

    """
    args = command.split(' ')
    for prog in args:
        if '=' not in prog:
            return prog
    return args[0]