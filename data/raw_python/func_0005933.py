def regenerate(session):
    """Regenerates header files for cmark under ./generated."""
    if platform.system() == 'Windows':
        output_dir = '../generated/windows'
    else:
        output_dir = '../generated/unix'

    session.run(shutil.rmtree, 'build', ignore_errors=True)
    session.run(os.makedirs, 'build')
    session.chdir('build')
    session.run('cmake', '../third_party/cmark')
    session.run(shutil.copy, 'src/cmark-gfm_export.h', output_dir)
    session.run(shutil.copy, 'src/cmark-gfm_version.h', output_dir)
    session.run(shutil.copy, 'src/config.h', output_dir)
    session.run(shutil.copy, 'extensions/cmark-gfm-extensions_export.h', output_dir)
    session.chdir('..')
    session.run(shutil.rmtree, 'build')