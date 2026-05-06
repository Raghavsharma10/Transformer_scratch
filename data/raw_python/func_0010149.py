def get_outfilename(url, domain=None):
    """Construct the output filename from domain and end of path."""
    if domain is None:
        domain = get_domain(url)

    path = '{url.path}'.format(url=urlparse(url))
    if '.' in path:
        tail_url = path.split('.')[-2]
    else:
        tail_url = path

    if tail_url:
        if '/' in tail_url:
            tail_pieces = [x for x in tail_url.split('/') if x]
            tail_url = tail_pieces[-1]

        # Keep length of return string below or equal to max_len
        max_len = 24
        if domain:
            max_len -= (len(domain) + 1)
        if len(tail_url) > max_len:
            if '-' in tail_url:
                tail_pieces = [x for x in tail_url.split('-') if x]
                tail_url = tail_pieces.pop(0)
                if len(tail_url) > max_len:
                    tail_url = tail_url[:max_len]
                else:
                    # Add as many tail pieces that can fit
                    tail_len = 0
                    for piece in tail_pieces:
                        tail_len += len(piece)
                        if tail_len <= max_len:
                            tail_url += '-' + piece
                        else:
                            break
            else:
                tail_url = tail_url[:max_len]

        if domain:
            return '{0}-{1}'.format(domain, tail_url).lower()
        return tail_url
    return domain.lower()