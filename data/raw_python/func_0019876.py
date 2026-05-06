def getVoIPchanStats(self, chantype, 
                         codec_list=('ulaw', 'alaw', 'gsm', 'g729')):
        """Query Asterisk Manager Interface for SIP / IAX2 Channel / Codec Stats.
        
        CLI Commands - sip show channels
                       iax2 show channnels
        
        @param chantype:   Must be 'sip' or 'iax2'.
        @param codec_list: List of codec names to parse.
                           (Codecs not in the list are summed up to the other 
                           count.)
        @return:           Dictionary of statistics counters for Active VoIP 
                           Channels.

        """
        chan = chantype.lower()
        if not self.hasChannelType(chan):
            return None
        if chan == 'iax2':
            cmd = "iax2 show channels"
        elif chan == 'sip':
            cmd = "sip show channels"
        else:
            raise AttributeError("Invalid channel type in query for Channel Stats.")
        cmdresp = self.executeCommand(cmd)
        lines = cmdresp.splitlines()
        headers = re.split('\s\s+', lines[0])
        try:
            idx = headers.index('Format')
        except ValueError:
            try:
                idx = headers.index('Form')
            except:
                raise Exception("Error in parsing header line of %s channel stats." 
                                % chan)
        codec_list = tuple(codec_list) + ('other', 'none')
        info_dict = dict([(k,0) for k in codec_list])
        for line in lines[1:-1]:
            codec = None
            cols = re.split('\s\s+', line)
            colcodec = cols[idx]
            mobj = re.match('0x\w+\s\((\w+)\)$', colcodec)
            if mobj:
                codec = mobj.group(1).lower()
            elif re.match('\w+$', colcodec):
                codec = colcodec.lower()
            if codec:
                if codec in info_dict:
                    info_dict[codec] += 1
                elif codec == 'nothing' or codec[0:4] == 'unkn':
                    info_dict['none'] += 1
                else:
                    info_dict['other'] += 1
        return info_dict