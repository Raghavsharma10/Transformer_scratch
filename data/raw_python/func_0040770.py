def init(ciprcfg, env, console):
    """
    Initialize a Corona project directory.
    """
    ciprcfg.create()

    templ_dir = path.join(env.skel_dir, 'default')

    console.quiet('Copying files from %s' % templ_dir)
    for src, dst in util.sync_dir_to(templ_dir, env.project_directory, ignore_existing=True):
        console.quiet('  %s -> %s' % (src, dst))

    src = path.join(env.code_dir, 'cipr.dev.lua')
    dst = path.join(env.project_directory, 'cipr.lua')
    console.quiet('  %s -> %s' % (src, dst))

    shutil.copy(src, dst)