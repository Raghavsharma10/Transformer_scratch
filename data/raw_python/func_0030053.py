def execute_command(cmd, execute, echo=True):
    """Execute a command in shell or just print it if execute is False"""
    if execute:
        if echo:
            print("Executing: " + cmd)
        return os.system(cmd)
    else:
        print(cmd)
        return 0