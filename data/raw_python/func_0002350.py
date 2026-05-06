def extract_srcset(self, srcset):
        """
        Handle ``srcset="image.png 1x, image@2x.jpg 2x"``
        """
        urls = []
        for item in srcset.split(','):
            if item:
                urls.append(unquote_utf8(item.rsplit(' ', 1)[0]))
        return urls