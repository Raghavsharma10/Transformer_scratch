def release():
    "check release before upload to PyPI"
    sh("paver bdist_wheel")
    wheels = path("dist").files("*.whl")
    if not wheels:
        error("\n*** ERROR: No release wheel was built!")
        sys.exit(1)
    if any(".dev" in i for i in wheels):
        error("\n*** ERROR: You're still using a 'dev' version!")
        sys.exit(1)

    # Check that source distribution can be built and is complete
    print('')
    print("~" * 78)
    print("TESTING SOURCE BUILD")
    sh( "{ command cd dist/ && unzip -q %s-%s.zip && command cd %s-%s/"
        "  && /usr/bin/python setup.py sdist >/dev/null"
        "  && if { unzip -ql ../%s-%s.zip; unzip -ql dist/%s-%s.zip; }"
        "        | cut -b26- | sort | uniq -c| egrep -v '^ +2 +' ; then"
        "       echo '^^^ Difference in file lists! ^^^'; false;"
        "    else true; fi; } 2>&1"
        % tuple([project["name"], version] * 4)
    )
    path("dist/%s-%s" % (project["name"], version)).rmtree()
    print("~" * 78)

    print('')
    print("Created", " ".join([str(i) for i in path("dist").listdir()]))
    print("Use 'paver sdist bdist_wheel' to build the release and")
    print("    'twine upload dist/*.{zip,whl}' to upload to PyPI")
    print("Use 'paver dist_docs' to prepare an API documentation upload")