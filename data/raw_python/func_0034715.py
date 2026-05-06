def _parse_posts(element):
    """
    Returns a list with posts.
    """

    posts = []
    items = element.findall("item")

    for item in items:
        title = item.find("./title").text
        link = item.find("./link").text
        pub_date = item.find("./pubDate").text
        creator = item.find("./{%s}creator" % DC_NAMESPACE).text
        guid = item.find("./guid").text
        description = item.find("./description").text
        content = item.find("./{%s}encoded" % CONTENT_NAMESPACE).text
        excerpt = item.find("./{%s}encoded" % EXCERPT_NAMESPACE).text
        post_id = item.find("./{%s}post_id" % WP_NAMESPACE).text
        post_date = item.find("./{%s}post_date" % WP_NAMESPACE).text
        post_date_gmt = item.find("./{%s}post_date_gmt" % WP_NAMESPACE).text
        status = item.find("./{%s}status" % WP_NAMESPACE).text
        post_parent = item.find("./{%s}post_parent" % WP_NAMESPACE).text
        menu_order = item.find("./{%s}menu_order" % WP_NAMESPACE).text
        post_type = item.find("./{%s}post_type" % WP_NAMESPACE).text
        post_name = item.find("./{%s}post_name" % WP_NAMESPACE).text
        is_sticky = item.find("./{%s}is_sticky" % WP_NAMESPACE).text
        ping_status = item.find("./{%s}ping_status" % WP_NAMESPACE).text
        post_password = item.find("./{%s}post_password" % WP_NAMESPACE).text
        category_items = item.findall("./category")

        categories = []
        tags = []

        for category_item in category_items:
            if category_item.attrib["domain"] == "category":
                item_list = categories
            else:
                item_list = tags

            item_list.append(category_item.attrib["nicename"])

        post = {
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "creator": creator,
            "guid": guid,
            "description": description,
            "content": content,
            "excerpt": excerpt,
            "post_id": post_id,
            "post_date": post_date,
            "post_date_gmt": post_date_gmt,
            "status": status,
            "post_parent": post_parent,
            "menu_order": menu_order,
            "post_type": post_type,
            "post_name": post_name,
            "categories": categories,
            "is_sticky": is_sticky,
            "ping_status": ping_status,
            "post_password": post_password,
            "tags": tags,
        }

        post["postmeta"] = _parse_postmeta(item)
        post["comments"] = _parse_comments(item)
        posts.append(post)

    return posts