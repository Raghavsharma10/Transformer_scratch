def exec_commands(commands: str, **parameters: Any) -> None:
    """Execute the given Python commands.

    Function |exec_commands| is thought for testing purposes only (see
    the main documentation on module |hyd|).  Seperate individual commands
    by semicolons and replaced whitespaces with underscores:

    >>> from hydpy.exe.commandtools import exec_commands
    >>> import sys
    >>> exec_commands("x_=_1+1;print(x)")
    Start to execute the commands ['x_=_1+1', 'print(x)'] for testing purposes.
    2

    |exec_commands| interprets double underscores as a single underscores:

    >>> exec_commands("x_=_1;print(x.____class____)")
    Start to execute the commands ['x_=_1', 'print(x.____class____)'] \
for testing purposes.
    <class 'int'>

    |exec_commands| evaluates additional keyword arguments before it
    executes the given commands:

    >>> exec_commands("e=x==y;print(e)", x=1, y=2)
    Start to execute the commands ['e=x==y', 'print(e)'] for testing purposes.
    False
    """
    cmdlist = commands.split(';')
    print(f'Start to execute the commands {cmdlist} for testing purposes.')
    for par, value in parameters.items():
        exec(f'{par} = {value}')
    for command in cmdlist:
        command = command.replace('__', 'temptemptemp')
        command = command.replace('_', ' ')
        command = command.replace('temptemptemp', '_')
        exec(command)