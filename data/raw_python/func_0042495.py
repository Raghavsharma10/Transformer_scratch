def from_env():
    """Get host/port settings from the environment."""
    if 'MICROMONGO_URI' in os.environ:
        return (os.environ['MICROMONGO_URI'],)
    host = os.environ.get('MICROMONGO_HOST', 'localhost')
    port = int(os.environ.get('MICROMONGO_PORT', 27017))
    return (host, port)