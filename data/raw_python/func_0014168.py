def urlEncodeAndJoin(self, seq, sepr=','):
        '''sepr.join(urlencode(seq))
        Args:
            seq: string list to be urlencoded
            sepr: join seq with sepr
        Returns:
            str
        '''
        try:
            from urllib.parse import quote_plus as encode
            return sepr.join([encode(x, encoding=CHARSET_UTF8) for x in seq])
        except ImportError:
            from urllib import quote as encode
            return sepr.join([i for i in map(lambda x: encode(x), seq)])