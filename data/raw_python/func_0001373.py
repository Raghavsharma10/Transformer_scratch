def login_defs():
    """Discover the minimum and maximum UID number."""
    uid_min = None
    uid_max = None
    login_defs_path = '/etc/login.defs'
    if os.path.exists(login_defs_path):
        with io.open(text_type(login_defs_path), encoding=text_type('utf-8')) as log_defs_file:
            login_data = log_defs_file.readlines()
        for line in login_data:
            if PY3:  # pragma: no cover
                line = str(line)
            if PY2:  # pragma: no cover
                line = line.encode(text_type('utf8'))
            if line[:7] == text_type('UID_MIN'):
                uid_min = int(line.split()[1].strip())
            if line[:7] == text_type('UID_MAX'):
                uid_max = int(line.split()[1].strip())
    if not uid_min:  # pragma: no cover
        uid_min = DEFAULT_UID_MIN
    if not uid_max:  # pragma: no cover
        uid_max = DEFAULT_UID_MAX
    return uid_min, uid_max