def install_docs(instance, clear_target):
    """Builds and installs the complete HFOS documentation."""

    _check_root()

    def make_docs():
        """Trigger a Sphinx make command to build the documentation."""
        log("Generating HTML documentation")

        try:
            build = Popen(
                [
                    'make',
                    'html'
                ],
                cwd='docs/'
            )

            build.wait()
        except Exception as e:
            log("Problem during documentation building: ", e, type(e),
                exc=True, lvl=error)
            return False
        return True

    make_docs()

    # If these need changes, make sure they are watertight and don't remove
    # wanted stuff!
    target = os.path.join('/var/lib/hfos', instance, 'frontend/docs')
    source = 'docs/build/html'

    log("Updating documentation directory:", target)

    if not os.path.exists(os.path.join(os.path.curdir, source)):
        log(
            "Documentation not existing yet. Run python setup.py "
            "build_sphinx first.", lvl=error)
        return

    if os.path.exists(target):
        log("Path already exists: " + target)
        if clear_target:
            log("Cleaning up " + target, lvl=warn)
            shutil.rmtree(target)

    log("Copying docs to " + target)
    copy_tree(source, target)
    log("Done: Install Docs")