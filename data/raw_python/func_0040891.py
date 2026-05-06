def _netid_category_url(netid, category_codes):
    """
    Return UWNetId resource for provided netid and category
    code or code list
    """
    return "{0}/{1}/category/{2}".format(
        url_base(), netid,
        (','.join([str(n) for n in category_codes])
         if isinstance(category_codes, (list, tuple))
         else category_codes))