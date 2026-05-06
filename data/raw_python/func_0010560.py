def initpeer(self, sock):
        '''
        Creates a new peer object for a nvalid socket and adds it to reactor's
        listen list
        '''
        location_json = requests.request("GET", "http://freegeoip.net/json/"
                                         + sock.getpeername()[0]).content
        location = json.loads(location_json)
        tpeer = peer.Peer(sock, self.reactor, self, location)
        self.peer_dict[sock] = tpeer
        self.reactor.select_list.append(tpeer)