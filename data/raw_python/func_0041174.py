def cleanup():
    """Clean up the installation directory."""
    lib_dir = os.path.join(os.environ['CONTAINER_DIR'], '_lib')
    if os.path.exists(lib_dir):
        shutil.rmtree(lib_dir)
    os.mkdir(lib_dir)