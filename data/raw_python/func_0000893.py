def title_prefix_json(soup):
    "titlePrefix with capitalisation changed"
    prefix = title_prefix(soup)
    prefix_rewritten = elifetools.json_rewrite.rewrite_json("title_prefix_json", soup, prefix)
    return prefix_rewritten