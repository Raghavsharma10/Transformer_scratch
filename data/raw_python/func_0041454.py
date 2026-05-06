def _clean_post_content(blog_url, content):
    """
    Replace import path with something relative to blog.
    """

    content = re.sub(
        "<img.src=\"%s(.*)\"" % blog_url,
        lambda s: "<img src=\"%s\"" % _get_relative_upload(s.groups(1)[0]),
        content)

    return content