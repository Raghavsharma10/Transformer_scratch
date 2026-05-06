def decensor_iter(posts_info: Iterable[dict], site_url: str = DEFAULT_SITE
                 ) -> Generator[dict, None, None]:
    """Apply decensoring on an iterable of posts info dicts from Danbooru API.
    Any censored post is automatically decensored if needed."""
    for info in posts_info:
        yield decensor(info, site_url)