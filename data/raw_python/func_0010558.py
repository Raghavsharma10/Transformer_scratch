def tracker_request(self):
        '''
        Sends the initial request to the tracker, compiling list of all peers
        announcing to the tracker
        '''

        assert self.torrent_dict['info']
        payload = self.build_payload()

        if self.torrent_dict['announce'].startswith('udp'):
            raise Exception('need to deal with UDP')

        else:
            self.r = requests.get(self.torrent_dict['announce'],
                                  params=payload)

        # Decoding response from tracker
        self.tracker_response = tparser.bdecode(self.r.content)
        self.get_peer_ips()