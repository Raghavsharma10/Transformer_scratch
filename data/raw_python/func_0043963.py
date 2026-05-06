def get_image(cls, url):
        """
        Returned Image instance has response url.
        This might be different than the url param because of redirects.
        """
        from PIL.ImageFile import Parser as PILParser

        length = 0
        raw_image = None
        with closing(request.get(url, stream=True)) as response:
            response.raise_for_status()
            response_url = response.url
            parser = PILParser()
            for chunk in response.iter_content(config.CHUNK_SIZE):
                length += len(chunk)
                if length > config.IMAGE_MAX_BYTESIZE:
                    del parser
                    raise cls.MaxBytesException
                parser.feed(chunk)
                # comment this to get the whole file
                if parser.image and parser.image.size:
                    raw_image = parser.image
                    del parser # free some memory
                    break
            # or this to get just the size and format
            # raw_image = parser.close()
        if length == 0:
            raise cls.ZeroBytesException
        if not raw_image:
            raise cls.NoImageException
        image = Image(response_url, raw_image.size, raw_image.format)
        return image