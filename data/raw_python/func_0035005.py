def cprint(message, status=None):
    """color printing based on status:

    None -> BRIGHT
    'ok' -> GREEN
    'err' -> RED
    'warn' -> YELLOW

    """
    # TODO use less obscure dict, probably "error", "warn", "success" as keys
    status = {'warn': Fore.YELLOW, 'err': Fore.RED,
              'ok': Fore.GREEN, None: Style.BRIGHT}[status]
    print(status + message + Style.RESET_ALL)