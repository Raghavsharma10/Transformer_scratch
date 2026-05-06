def make_dist(toxinidir, toxdir, package):
    """zip up the package into the toxdir."""
    dist = os.path.join(toxdir, "dist")
    # Suppress warnings.
    success = safe_shell_out(["python", "setup.py", "sdist", "--quiet",
                              "--formats=zip", "--dist-dir", dist],
                             cwd=toxinidir)
    if success:
        return os.path.join(dist, package + ".zip")