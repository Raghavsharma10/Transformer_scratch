def install_reqs(venv, repo_dest):
    """Installs all compiled requirements that can't be shipped in vendor."""
    with dir_path(repo_dest):
        args = ['-r', 'requirements/compiled.txt']
        if not verbose:
            args.insert(0, '-q')
        subprocess.check_call([os.path.join(venv, 'bin', 'pip'), 'install'] +
                              args)