def meson_setup():
    """
    attempt to build with Meson + Ninja
    """
    meson_exe = shutil.which('meson')
    ninja_exe = shutil.which('ninja')

    if not meson_exe or not ninja_exe:
        raise FileNotFoundError('Meson or Ninja not available')

    if not (BINDIR / 'build.ninja').is_file():
        subprocess.check_call([meson_exe, str(SRCDIR)], cwd=BINDIR)

    ret = subprocess.run(ninja_exe, cwd=BINDIR, stderr=subprocess.PIPE,
                         universal_newlines=True)

    result(ret)