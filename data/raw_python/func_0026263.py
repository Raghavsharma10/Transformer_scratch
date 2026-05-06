def cmake_setup():
    """
    attempt to build using CMake >= 3
    """
    cmake_exe = shutil.which('cmake')
    if not cmake_exe:
        raise FileNotFoundError('CMake not available')

    wopts = ['-G', 'MinGW Makefiles', '-DCMAKE_SH="CMAKE_SH-NOTFOUND'] if os.name == 'nt' else []

    subprocess.check_call([cmake_exe] + wopts + [str(SRCDIR)],
                          cwd=BINDIR)

    ret = subprocess.run([cmake_exe, '--build', str(BINDIR)],
                         stderr=subprocess.PIPE,
                         universal_newlines=True)

    result(ret)