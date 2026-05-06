def debug_process(pid):
    """Interrupt a running process and debug it."""
    os.kill(pid, signal.SIGUSR1)  # Signal process.
    pipe = NamedPipe(pipename(pid), 1)
    try:
        while pipe.is_open():
            txt=raw_input(pipe.get()) + '\n'
            pipe.put(txt)
    except EOFError:
        pass # Exit.
    pipe.close()