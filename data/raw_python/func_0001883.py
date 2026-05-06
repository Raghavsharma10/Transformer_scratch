def get_pages(url):
    """
    Return the 'pages' from the starting url
    Technically, look for the 'next 50' link, yield and download it,  repeat
    """
    while True:
        yield url
        doc = html.parse(url).find("body")
        links = [a for a in doc.findall(".//a") if a.text and a.text.startswith("next ")]
        if not links:
            break
        url = urljoin(url, links[0].get('href'))