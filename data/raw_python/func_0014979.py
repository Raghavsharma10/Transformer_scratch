def generate_py2k(config, py2k_dir=PY2K_DIR, run_tests=False):
    """Generate Python 2 code from Python 3 code."""
    def copy(src, dst):
        if (not os.path.isfile(dst) or
                os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copy(src, dst)
            return dst
        return None

    def copy_data(src, dst):
        if (not os.path.isfile(dst) or
                os.path.getmtime(src) > os.path.getmtime(dst) or
                os.path.getsize(src) != os.path.getsize(dst)):
            shutil.copy(src, dst)
            return dst
        return None

    copied_py_files = []
    test_scripts = []

    if not os.path.isdir(py2k_dir):
        os.makedirs(py2k_dir)

    packages_root = get_cfg_value(config, 'files', 'packages_root')

    for name in get_cfg_value(config, 'files', 'packages'):
        name = name.replace('.', os.path.sep)
        py3k_path = os.path.join(packages_root, name)
        py2k_path = os.path.join(py2k_dir, py3k_path)
        if not os.path.isdir(py2k_path):
            os.makedirs(py2k_path)
        for fn in os.listdir(py3k_path):
            path = os.path.join(py3k_path, fn)
            if not os.path.isfile(path):
                continue
            if not os.path.splitext(path)[1].lower() == '.py':
                continue
            new_path = os.path.join(py2k_path, fn)
            if copy(path, new_path):
                copied_py_files.append(new_path)

    for name in get_cfg_value(config, 'files', 'modules'):
        name = name.replace('.', os.path.sep) + '.py'
        py3k_path = os.path.join(packages_root, name)
        py2k_path = os.path.join(py2k_dir, py3k_path)
        dirname = os.path.dirname(py2k_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if copy(py3k_path, py2k_path):
            copied_py_files.append(py2k_path)

    for name in get_cfg_value(config, 'files', 'scripts'):
        py3k_path = os.path.join(packages_root, name)
        py2k_path = os.path.join(py2k_dir, py3k_path)
        dirname = os.path.dirname(py2k_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if copy(py3k_path, py2k_path):
            copied_py_files.append(py2k_path)

    setup_py_path = os.path.abspath(__file__)

    for pattern in get_cfg_value(config, 'files', 'extra_files'):
        for path in glob.glob(pattern):
            if os.path.abspath(path) == setup_py_path:
                continue
            py2k_path = os.path.join(py2k_dir, path)
            py2k_dirname = os.path.dirname(py2k_path)
            if not os.path.isdir(py2k_dirname):
                os.makedirs(py2k_dirname)
            filename = os.path.split(path)[1]
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.py':
                if copy(path, py2k_path):
                    copied_py_files.append(py2k_path)
            else:
                copy_data(path, py2k_path)
            if (os.access(py2k_path, os.X_OK) and
                    re.search(r"\btest\b|_test\b|\btest_", filename)):
                test_scripts.append(py2k_path)

    for package, patterns in get_package_data(
            get_cfg_value(config, 'files', 'package_data')).items():
        for pattern in patterns:
            py3k_pattern = os.path.join(packages_root, package, pattern)
            for py3k_path in glob.glob(py3k_pattern):
                py2k_path = os.path.join(py2k_dir, py3k_path)
                py2k_dirname = os.path.dirname(py2k_path)
                if not os.path.isdir(py2k_dirname):
                    os.makedirs(py2k_dirname)
                copy_data(py3k_path, py2k_path)

    if copied_py_files:
        copied_py_files.sort()
        try:
            run_3to2(copied_py_files)
            write_py2k_header(copied_py_files)
        except:
            shutil.rmtree(py2k_dir)
            raise

    if run_tests:
        for script in test_scripts:
            subprocess.check_call([script])