def reload_me(*args, ignore_patterns=[]):
    """Reload currently running command with given args"""
    
    command = [sys.executable, sys.argv[0]]
    command.extend(args)

    reload(*command, ignore_patterns=ignore_patterns)