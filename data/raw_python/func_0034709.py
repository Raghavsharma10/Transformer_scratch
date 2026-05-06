def parse(path):
    """
    Parses xml and returns a formatted dict.

    Example:

        wpparser.parse("./blog.wordpress.2014-09-26.xml")

    Will return:

        {
        "blog": {
            "tagline": "Tagline",
            "site_url": "http://marteinn.se/blog",
            "blog_url": "http://marteinn.se/blog",
            "language": "en-US",
            "title": "Marteinn / Blog"
        },
        "authors: [{
            "login": "admin",
            "last_name": None,
            "display_name": "admin",
            "email": "martin@marteinn.se",
            "first_name": None}
        ],
        "categories": [{
            "parent": None,
            "term_id": "3",
            "name": "Action Script",
            "nicename": "action-script",
            "children": [{
                "parent": "action-script",
                "term_id": "20",
                "name": "Flash related",
                "nicename": "flash-related",
                "children": []
            }]
        }],
        "tags": [{"term_id": "36", "slug": "bash", "name": "Bash"}],
        "posts": [{
            "creator": "admin",
            "excerpt": None,
            "post_date_gmt": "2014-09-22 20:10:40",
            "post_date": "2014-09-22 21:10:40",
            "post_type": "post",
            "menu_order": "0",
            "guid": "http://marteinn.se/blog/?p=828",
            "title": "Post Title",
            "comments": [{
                "date_gmt": "2014-09-24 23:08:31",
                "parent": "0",
                "date": "2014-09-25 00:08:31",
                "id": "85929",
                "user_id": "0",
                "author": u"Author",
                "author_email": None,
                "author_ip": "111.111.111.111",
                "approved": "1",
                "content": u"Comment title",
                "author_url": "http://example.com",
                "type": "pingback"
            }],
            "content": "Text",
            "post_parent": "0",
            "post_password": None,
            "status": "publish",
            "description": None,
            "tags": ["tag"],
            "ping_status": "open",
            "post_id": "828",
            "link": "http://www.marteinn.se/blog/slug/",
            "pub_date": "Mon, 22 Sep 2014 20:10:40 +0000",
            "categories": ["category"],
            "is_sticky": "0",
            "post_name": "slug"
        }]
        }
    """

    doc = ET.parse(path).getroot()

    channel = doc.find("./channel")

    blog = _parse_blog(channel)
    authors = _parse_authors(channel)
    categories = _parse_categories(channel)
    tags = _parse_tags(channel)
    posts = _parse_posts(channel)

    return {
        "blog": blog,
        "authors": authors,
        "categories": categories,
        "tags": tags,
        "posts": posts,
    }