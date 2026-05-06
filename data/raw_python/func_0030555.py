def create_mirror_settings(repo_url):
    """
    Creates settings.xml in current working directory, which when used makes Maven use given repo URL as a mirror of all
    repositories to look at.

    :param repo_url: the repository URL to use
    :returns: filepath to the created file
    """
    cwd = os.getcwd()
    settings_path = os.path.join(cwd, "settings.xml")

    settings_file = None
    try:
        settings_file = open(settings_path, "w")
        settings_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        settings_file.write('<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"\n')
        settings_file.write('          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        settings_file.write('          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd">\n')
        settings_file.write('<mirrors>\n')
        settings_file.write('    <mirror>\n')
        settings_file.write('      <id>repo-mirror</id>\n')
        settings_file.write('        <url>%s</url>\n' % repo_url)
        settings_file.write('      <mirrorOf>*</mirrorOf>\n')
        settings_file.write('    </mirror>\n')
        settings_file.write('  </mirrors>\n')
        settings_file.write('</settings>\n')
    finally:
        if settings_file:
            settings_file.close()

    return settings_path