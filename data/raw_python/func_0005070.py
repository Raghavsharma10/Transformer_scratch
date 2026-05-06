def encoded_words_to_text(self, in_encoded_words: str):
        """Pull out the character set, encoding, and encoded text from the input
        encoded words. Next, it decodes the encoded words into a byte string,
        using either the quopri module or base64 module as determined by the
        encoding. Finally, it decodes the byte string using the
        character set and returns the result.

        See:

        - https://github.com/isogeo/isogeo-api-py-minsdk/issues/32
        - https://dmorgan.info/posts/encoded-word-syntax/

        :param str in_encoded_words: base64 or quori encoded character string.
        """
        # handle RFC2047 quoting
        if '"' in in_encoded_words:
            in_encoded_words = in_encoded_words.strip('"')
        # regex
        encoded_word_regex = r"=\?{1}(.+)\?{1}([B|Q])\?{1}(.+)\?{1}="
        # pull out
        try:
            charset, encoding, encoded_text = re.match(
                encoded_word_regex, in_encoded_words
            ).groups()
        except AttributeError:
            logging.debug("Input text was not encoded into base64 or quori")
            return in_encoded_words

        # decode depending on encoding
        if encoding == "B":
            byte_string = base64.b64decode(encoded_text)
        elif encoding == "Q":
            byte_string = quopri.decodestring(encoded_text)
        return byte_string.decode(charset)