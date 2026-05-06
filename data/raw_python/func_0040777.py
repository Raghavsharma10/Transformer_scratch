def packageipa(env, console):
    """
    Package the built app as an ipa for distribution in iOS App Store
    """
    ipa_path, app_path = _get_ipa(env)
    output_dir = path.dirname(ipa_path)

    if path.exists(ipa_path):
        console.quiet('Removing %s' % ipa_path)
        os.remove(ipa_path)

    zf = zipfile.ZipFile(ipa_path, mode='w')
    payload_dir = 'Payload'

    for (dirpath, dirnames, filenames) in os.walk(app_path):
        for filename in filenames:
            filepath = path.join(dirpath, filename)
            prefix = path.commonprefix([filepath, path.dirname(app_path)])
            write_path = path.join(payload_dir, filepath[len(prefix) + 1:])

            console.quiet('Write %s' % write_path)

            zf.write(filepath, write_path)

    zf.close()

    console.quiet('Packaged %s' % ipa_path)