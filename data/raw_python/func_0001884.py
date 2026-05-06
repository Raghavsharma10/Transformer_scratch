def get_article_urls(url):
    """
    Return the articles from a page
    Technically, look for a div with class mw-search-result-heading
    and get the first link from this div
    """
    doc = html.parse(url).getroot()
    for div in doc.cssselect("div.mw-search-result-heading"):
        href = div.cssselect("a")[0].get('href')
        if ":" in href:
            continue # skip Category: links
        href = urljoin(url, href)
        yield href