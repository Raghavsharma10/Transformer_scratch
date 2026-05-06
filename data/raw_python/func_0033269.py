def gen_headers() -> Dict[str, str]:
    """Generate a header pairing."""
    ua_list: List[str] = ['Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.117 Safari/537.36']
    headers: Dict[str, str] = {'User-Agent': ua_list[random.randint(0, len(ua_list) - 1)]}
    return headers