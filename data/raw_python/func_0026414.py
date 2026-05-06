def uninstall():
    """Uninstall data and resource locations"""

    _check_root()

    response = _ask("This will delete all data of your HFOS installations! Type"
                    "YES to continue:", default="N", show_hint=False)
    if response == 'YES':
        shutil.rmtree('/var/lib/hfos')
        shutil.rmtree('/var/cache/hfos')