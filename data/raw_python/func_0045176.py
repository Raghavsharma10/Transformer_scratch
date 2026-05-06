def get_broadcast_date(pid):
    """Take BBC pid (string); extract and return broadcast date as string."""
    print("Extracting first broadcast date...")
    broadcast_etree = open_listing_page(pid + '/broadcasts.inc')
    original_broadcast_date, = broadcast_etree.xpath(
        '(//div[@class="grid__inner"]//div'
        '[@class="broadcast-event__time beta"]/@title)[1]')
    return original_broadcast_date