def import_blogger(filepath):
    """Imports A Blogger export and converts it to a Blended site"""

    print("\nBlended: Static Website Generator -\n")

    checkConfig()
    print("Importing from Blogger...")

    blogger = parseXML(filepath)

    wname = blogger.feed.title.cdata
    aname = blogger.feed.author.name.cdata.strip()

    createBlendedFolders()

    # Populate the configuration file
    createConfig(app_version=app_version, wname=wname, aname=aname)

    for entry in blogger.feed.entry:
        if "post" in entry.id.cdata:
            with open(os.path.join(cwd, "content", entry.title.cdata.replace(" ", "_") + ".html"), 'w') as wfile:
                wfile.write(entry.content.cdata.strip())

    print("\nYour website has been imported from Blogger.")