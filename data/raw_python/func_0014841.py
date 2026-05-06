def config(host, seq, option, value):
    """Set configuration parameters of the drone."""
    at(host, 'CONFIG', seq, [str(option), str(value)])