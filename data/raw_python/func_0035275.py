def clean():
    "take out the trash"
    src_dir = easy.options.setdefault("docs", {}).get('src_dir', None)
    if src_dir is None:
        src_dir = 'src' if easy.path('src').exists() else '.'

    with easy.pushd(src_dir):
        for pkg in set(easy.options.setup.packages) | set(("tests",)):
            for filename in glob.glob(pkg.replace('.', os.sep) + "/*.py[oc~]"):
                easy.path(filename).remove()