def build_rss_feed(podcast):
    """
    Builds a podcast RSS feed and returns an xml file.

    :param podcast:
        A Podcast model to build the RSS feed from.
    """
    if not os.path.exists(podcast.output_path):
        os.makedirs(podcast.output_path)

    rss = ET.Element('rss', attrib={'xmlns:itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd', 'version': '2.0'})

    channel = ET.SubElement(rss, 'channel')
    ET.SubElement(channel, 'title').text = podcast.title
    ET.SubElement(channel, 'link').text = podcast.link
    ET.SubElement(channel, 'copyright').text = podcast.copyright
    ET.SubElement(channel, 'itunes:subtitle').text = podcast.subtitle
    ET.SubElement(channel, 'itunes:author').text = podcast.author
    ET.SubElement(channel, 'itunes:summary').text = podcast.description
    ET.SubElement(channel, 'description').text = podcast.description

    owner = ET.SubElement(channel, 'itunes:owner')
    ET.SubElement(owner, 'itunes:name').text = podcast.owner_name
    ET.SubElement(owner, 'itunes:email').text = podcast.owner_email

    ET.SubElement(channel, 'itunes:image').text = podcast.image

    for category in podcast.categories:
        ET.SubElement(channel, 'itunes:category').text = category

    for episode in sorted(podcast.episodes.values(), key=lambda x: x.publish_date):
        if episode.published is True:
            item = ET.SubElement(channel, 'item')
            ET.SubElement(item, 'title').text = episode.title
            ET.SubElement(item, 'author').text = episode.author
            ET.SubElement(item, 'summary').text = episode.summary
            ET.SubElement(item, 'enclosure', attrib={'url': podcast.link + '/' + episode.link, 'length': str(episode.length), 'type': 'audio/x-mp3'})
            ET.SubElement(item, 'guid').text = podcast.link + '/' + episode.link
            ET.SubElement(item, 'pubDate').text = episode.publish_date.strftime('%a, %d %b %Y %H:%M:%S UTC')
            ET.SubElement(item, 'itunes:duration').text = episode.duration

    tree = ET.ElementTree(rss)
    with open(podcast.output_path + '/feed.xml', 'wb') as feed:
        tree.write(feed)