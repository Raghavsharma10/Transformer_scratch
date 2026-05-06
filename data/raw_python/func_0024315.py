def defaultCrawlId():
    """
    Provide a reasonable default crawl name using the user name and date
    """

    timestamp = datetime.now().isoformat().replace(':', '_')
    user = getuser()
    return '_'.join(('crawl', user, timestamp))