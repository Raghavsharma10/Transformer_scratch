def extract_listing(pid):
    """Extract listing; return list of tuples (artist(s), title, label)."""
    print("Extracting tracklisting...")
    listing_etree = open_listing_page(pid + '/segments.inc')
    track_divs = listing_etree.xpath('//div[@class="segment__track"]')

    listing = []
    for track_div in track_divs:
        try:
            artist_names = track_div.xpath('.//span[@property="byArtist"]'
                                           '//span[@class="artist"]/text()')
        except ValueError:
            artist_names = ['']

        if not artist_names:
            artist_names = ['']

        if len(artist_names) > 1:
            artists = ', '.join(artist_names[:-1]) + ' & ' + artist_names[-1]
        else:
            artists = artist_names[0]

        try:
            title, = track_div.xpath('.//p/span[@property="name"]/text()')
        except ValueError:
            title = ''

        try:
            label, = track_div.xpath('.//abbr[@title="Record Label"]'
                                     '/span[@property="name"]/text()')
        except ValueError:
            label = ''
        listing.append((artists, title, label))
    return listing