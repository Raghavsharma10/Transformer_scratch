def interactive(outdir):
    """Blends the generated files and outputs a HTML website on file change"""

    print("Building your Blended files into a website!")

    global outdir_type
    outdir_type = outdir

    reload(sys)
    sys.setdefaultencoding('utf8')

    build_files(outdir)

    print("Watching the content and templates directories for changes, press CTRL+C to stop...\n")

    w = Watcher()
    w.run()