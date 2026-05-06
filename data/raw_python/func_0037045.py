def urlopen(link):
    """Return urllib2 urlopen
    """
    try:
        return urllib2.urlopen(link)
    except urllib2.URLError:
        pass
    except ValueError:
        return ""
    except KeyboardInterrupt:
        print("")
        raise SystemExit()