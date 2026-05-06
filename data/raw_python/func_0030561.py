def get_lyric_by_songid(self, mid):
        """Get song lyric
        :param mid: music id
        :return: {
            lrc: {
                version: int,
                lyric: str
            },
            tlyric: {
                version: int,
                lyric: str
            }
            sgc: bool,
            qfy: bool,
            sfy: bool,
            transUser: {},
            code: int,
        }
        """
        # tv 表示翻译。-1：表示要翻译，1：不要
        url = uri + '/song/lyric?' + 'id=' + str(mid) + '&lv=1&kv=1&tv=-1'
        return self.request('GET', url)