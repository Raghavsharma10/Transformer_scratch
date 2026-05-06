def build(env, ciprcfg, console):
    """
    Build the current project for distribution
    """
    os.putenv('CIPR_PACKAGES', env.package_dir)
    os.putenv('CIPR_PROJECT', env.project_directory)

    build_settings = path.join(env.project_directory, 'build.settings')

    with open(build_settings, 'r') as f:
        data = f.read()

    m = _build_re.search(data)
    if m:
        ver = int(m.group(2))
        data = data.replace(m.group(0), 'CFBundleVersion = "%d"' % (ver + 1))

        with open(build_settings, 'w') as f:
            f.write(data)


    if path.exists(env.build_dir):
        shutil.rmtree(env.build_dir)

    os.makedirs(env.build_dir)

    if path.exists(env.dist_dir):
        shutil.rmtree(env.dist_dir)

    os.makedirs(env.dist_dir)

    console.normal('Building in %s' % env.build_dir)

    console.normal('Copy project files...')
    for src, dst in util.sync_dir_to(env.project_directory, env.build_dir, exclude=['.cipr', '.git', 'build', 'dist', '.*']):
        console.quiet('  %s -> %s' % (src, dst))
        if src.endswith('.lua'):
            _fix_lua_module_name(src, dst)


    console.normal('Copy cipr packages...')
    for package in ciprcfg.packages.keys():
        for src, dst in util.sync_lua_dir_to(path.join(env.package_dir, package), env.build_dir, exclude=['.git'], include=['*.lua']):
            console.quiet('  %s -> %s' % (src, dst))
            if src.endswith('.lua'):
                _fix_lua_module_name(src, dst)

    src = path.join(env.code_dir, 'cipr.lua')
    dst = path.join(env.build_dir, 'cipr.lua')
    shutil.copy(src, dst)

    cmd = AND(clom.cd(env.build_dir), clom[CORONA_SIMULATOR_PATH](env.build_dir))

    console.normal('Be sure to output your app to %s' % env.dist_dir)

    try:
        cmd.shell.execute()
    except KeyboardInterrupt:
        pass