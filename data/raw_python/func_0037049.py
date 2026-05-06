def mirror():
    """Get mirror from slackpkg mirrors file
    """
    slack_mirror = read_config(
        read_file("{0}{1}".format(etc_slackpkg, "mirrors")))
    if slack_mirror:
        return slack_mirror + changelog_txt
    else:
        print("\nYou do not have any mirror selected in /etc/slackpkg/mirrors"
              "\nPlease edit that file and uncomment ONE mirror.\n")
        return ""