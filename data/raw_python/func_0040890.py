def update_catagory(netid, category_code, status):
    """
    Post a subscriptionfor the given netid
    and category_code
    """
    url = "{0}/category".format(url_version())
    body = {
        "categoryCode": category_code,
        "status": status,
        "categoryList": [{"netid": netid}]
    }

    response = post_resource(url, json.dumps(body))
    return json.loads(response)