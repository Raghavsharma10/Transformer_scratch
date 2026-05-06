def start_console(local_vars={}):
    '''Starts a console; modified from code.interact'''
    transforms.CONSOLE_ACTIVE = True
    transforms.remove_not_allowed_in_console()
    sys.ps1 = prompt
    console = ExperimentalInteractiveConsole(locals=local_vars)
    console.interact(banner=banner)