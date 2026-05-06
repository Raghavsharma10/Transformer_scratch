def from_dict(raw_data):
        """Create Image from raw dictionary data."""
        url = None
        width = None
        height = None
        try:
            url = raw_data['url']
            width = raw_data['width']
            height = raw_data['height']
        except KeyError:
            raise ValueError('Unexpected image json structure')
        except TypeError:
            # Happens when raw_data is None, i.e. when a term has no image:
            pass
        return Image(url, width, height)