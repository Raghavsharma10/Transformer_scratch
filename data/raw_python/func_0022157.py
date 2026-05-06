def url_decode(url):
        """
        URL Decode function using REGEX

        **Parameters:**

          - **url:** URLENCODED text string

        **Returns:** Non URLENCODED string
        """
        return re.compile('%([0-9a-fA-F]{2})', re.M).sub(lambda m: chr(int(m.group(1), 16)), url)