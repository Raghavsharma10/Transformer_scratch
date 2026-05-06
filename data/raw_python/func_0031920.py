def clean_built(outdir):
    """Removes all built files"""
    print("Removing the built files!")

    # Remove the  build folder
    build_dir = os.path.join(cwd, outdir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)