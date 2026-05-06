def import_wp(filepath):
    """Imports A WordPress export and converts it to a Blended site"""

    print("\nBlended: Static Website Generator -\n")

    checkConfig()
    print("Importing from WordPress...")

    wp = parseXML(filepath)

    wname = wp.rss.channel.title.cdata
    wdesc = wp.rss.channel.description.cdata
    wlan = wp.rss.channel.language.cdata
    wurl = wp.rss.channel.link.cdata
    aname = wp.rss.channel.wp_author.wp_author_display_name.cdata.strip()

    createBlendedFolders()

    # Populate the configuration file
    createConfig(app_version=app_version, wname=wname,
                 wdesc=wdesc, wlan=wlan, wurl=wurl, aname=aname)

    for item in wp.rss.channel.item:
        with open(os.path.join(cwd, "content", item.title.cdata.replace(" ", "_") + ".html"), 'w') as wfile:
            wfile.write(item.content_encoded.cdata.strip())

    print("\nYour website has been imported from WordPress.")