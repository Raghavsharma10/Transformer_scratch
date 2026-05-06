def get_public_cms_page_urls(*, language_code):
    """
    :param language_code: e.g.: "en" or "de"
    :return: Tuple with all public urls in the given language
    """
    pages = Page.objects.public()
    urls = [page.get_absolute_url(language=language_code) for page in pages]
    urls.sort()
    return tuple(urls)