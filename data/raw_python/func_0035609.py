def fake_upload_from_url(url):
    """ Return a 'fake' upload data record, so that upload errors
        can be mitigated by using an original / alternative URL,
        especially when cross-loading from the web.
    """
    return parts.Bunch(
        image=parts.Bunch(
            animated='false', bandwidth=0, caption=None, views=0, deletehash=None, hash=None,
            name=(url.rsplit('/', 1) + [url])[1], title=None, type='image/*', width=0, height=0, size=0,
            datetime=int(time.time()), # XXX was fmt.iso_datetime() - in API v2 this is a UNIX timestamp
            id='xxxxxxx', link=url, account_id=0, account_url=None, ad_type=0, ad_url='',
            description=None, favorite=False, in_gallery=False, in_most_viral=False,
            is_ad=False, nsfw=None, section=None, tags=[], vote=None,
        ),
        links=parts.Bunch(
            delete_page=None, imgur_page=None,
            original=url, large_thumbnail=url, small_square=url,
        ))