def load_id3(self, track):
        """ Load id3 tags from strack metadata """
        if not isinstance(track, strack):
            raise TypeError('strack object required')

        timestamp = calendar.timegm(parse(track.get("created-at")).timetuple())

        self.mapper[TIT1] = TIT1(text=track.get("description"))
        self.mapper[TIT2] = TIT2(text=track.get("title"))
        self.mapper[TIT3] = TIT3(text=track.get("tags-list"))
        self.mapper[TDOR] = TDOR(text=str(timestamp))
        self.mapper[TLEN] = TLEN(text=track.get("duration"))
        self.mapper[TOFN] = TOFN(text=track.get("permalink"))
        self.mapper[TCON] = TCON(text=track.get("genre"))
        self.mapper[TCOP] = TCOP(text=track.get("license"))
        self.mapper[WOAS] = WOAS(url=track.get("permalink-url"))
        self.mapper[WOAF] = WOAF(url=track.get("uri"))
        self.mapper[TPUB] = TPUB(text=track.get("username"))
        self.mapper[WOAR] = WOAR(url=track.get("user-url"))
        self.mapper[TPE1] = TPE1(text=track.get("artist"))
        self.mapper[TALB] = TALB(text="%s Soundcloud tracks"
                                 % track.get("artist"))

        if track.get("artwork-path") is not None:
            self.mapper[APIC] = APIC(value=track.get("artwork-path"))