def _parse_blog(element):
    """
    Parse and return genral blog data (title, tagline etc).
    """

    title = element.find("./title").text
    tagline = element.find("./description").text
    language = element.find("./language").text
    site_url = element.find("./{%s}base_site_url" % WP_NAMESPACE).text
    blog_url = element.find("./{%s}base_blog_url" % WP_NAMESPACE).text

    return {
        "title": title,
        "tagline": tagline,
        "language": language,
        "site_url": site_url,
        "blog_url": blog_url,
    }