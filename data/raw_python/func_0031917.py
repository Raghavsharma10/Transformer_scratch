def init():
    """Initiates a new website"""

    print("Blended: Static Website Generator -\n")

    checkConfig()

    if (sys.version_info > (3, 0)):
        wname = input("Website Name: ")
        wdesc = input("Website Description: ")
        wlan = input("Website Language: ")
        wlic = input("Website License: ")
        aname = input("Author(s) Name(s): ")
    else:
        wname = raw_input("Website Name: ")
        wdesc = raw_input("Website Description: ")
        wlan = raw_input("Website Language: ")
        wlic = raw_input("Website License: ")
        aname = raw_input("Author(s) Name(s): ")

    createBlendedFolders()

    # Populate the configuration file
    createConfig(app_version=app_version, wname=wname,
                 wdesc=wdesc, wlic=wlic, wlan=wlan, aname=aname)

    print("\nThe required files for your website have been generated.")