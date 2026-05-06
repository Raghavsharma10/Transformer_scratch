def presets_dir():
    """Return presets directory"""
    default_presets_dir = os.path.join(
        os.path.expanduser("~"), ".be", "presets")
    presets_dir = os.environ.get(BE_PRESETSDIR) or default_presets_dir
    if not os.path.exists(presets_dir):
        os.makedirs(presets_dir)
    return presets_dir