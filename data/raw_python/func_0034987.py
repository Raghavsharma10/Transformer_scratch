def dist_docs():
    "create a documentation bundle"
    dist_dir = path("dist")
    docs_package = path("%s/%s-%s-docs.zip" % (dist_dir.abspath(), options.setup.name, options.setup.version))

    dist_dir.exists() or dist_dir.makedirs()
    docs_package.exists() and docs_package.remove()

    sh(r'cd build/apidocs && zip -qr9 %s .' % (docs_package,))

    print('')
    print("Upload @ http://pypi.python.org/pypi?:action=pkg_edit&name=%s" % ( options.setup.name,))
    print(docs_package)