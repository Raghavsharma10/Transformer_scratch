def scrape_wikinews(conn, project, articleset, query):
    """
    Scrape wikinews articles from the given query
    @param conn: The AmcatAPI object
    @param articleset: The target articleset ID
    @param category: The wikinews category name
    """
    url = "http://en.wikinews.org/w/index.php?search={}&limit=50".format(query)
    logging.info(url)
    for page in get_pages(url):
        urls = get_article_urls(page)
        arts = list(get_articles(urls))
        logging.info("Adding {} articles to set {}:{}"
                     .format(len(arts), project, articleset))
        conn.create_articles(project=project, articleset=articleset,
                            json_data=arts)