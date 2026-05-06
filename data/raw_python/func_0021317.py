def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)