def expanddotpaths(env, console):
    """
    Move files with dots in them to sub-directories
    """
    for filepath in os.listdir(path.join(env.dir)):
        filename, ext = path.splitext(filepath)
        if ext == '.lua' and '.' in filename:
            paths, newfilename = filename.rsplit('.', 1)
            newpath = paths.replace('.', '/')
            newfilename = path.join(newpath, newfilename) + ext

            console.quiet('Move %s to %s' % (filepath, newfilename))

            fullpath = path.join(env.project_directory, newpath)
            if not path.exists(fullpath):
                os.makedirs(fullpath)

            clom.git.mv(filepath, newfilename).shell.execute()