def extract_entry(scraped_info):
        """
        Transform scraped_info dictionary into an entry, under the assumption that there is only
        one track in 'track' list, since each video/audio is instantiated individually
        on the RMF website and each of them is scraped independently, so there shouldn't be cases
        when there are 2 unrelated tracks in one info_dict.

        Args:
            scraped_info (dict): Video info dict, scraped straight from the website.

        Returns:
            dict: Entry containing title, formats (url, quality), thumbnail, etc.

        """
        quality_mapping = {  # ascending in terms of quality
            'lo': 0,
            'hi': 1
        }

        entry = scraped_info['tracks'][0]
        '''
        The structure of entry is as follows:

        'src': {
            'hi': [
                {
                'src': 'http://v.iplsc.com/30-11-gosc-marek-jakubiak/0007124B3CGCAE6P-A1.mp4',
                'type': 'video/mp4'
                }
            ],
            'lo': [
                {
                'src': 'http://v.iplsc.com/30-11-gosc-marek-jakubiak/0007124B3CGCAE6P-A1.mp4',
                'type': 'video/mp4'
                }
            ]
        }
        '''

        sources = entry.pop('src')

        # TODO: #LOW_PRIOR Remove date from title of audio files e.g. '10.06 Gość: Jarosław Gowin'

        formats = []
        for src_name, src in sources.items():
            url = src[0]['src']
            formats.append({
                'url': url,
                'quality': quality_mapping[src_name],
                'ext': get_ext(url),
                'width': int(scraped_info.get('width', 0)),
                'height': int(scraped_info.get('height', 0)),
            })

        # outer level url and ext come from the video of the lowest quality
        # you can access rest of the urls under 'formats' key
        worst_format = min(formats, key=lambda f: f['quality'])
        entry.update({
            **entry.pop('data'),
            'formats': formats,
            'url': worst_format['url'],
            'ext': worst_format['ext']
        })

        return entry