def _make_links_absolute(html, base_url):
    """
    Make all links absolute.
    """
    url_changes = []

    soup = BeautifulSoup(html)
    for tag in soup.find_all('a', href=True):
        old = tag['href']
        fixed = urljoin(base_url, old)
        if old != fixed:
            url_changes.append((old, fixed))
            tag['href'] = fixed

    for tag in soup.find_all('img', src=True):
        old = tag['src']
        fixed = urljoin(base_url, old)
        if old != fixed:
            url_changes.append((old, fixed))
            tag['src'] = fixed

    return mark_safe(six.text_type(soup)), url_changes