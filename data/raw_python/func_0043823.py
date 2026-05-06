def requestJoinDetails(self):
        """add configuration information to the SC2 protocol join request
    REQUIREMENTS FOR SUCCESSFUL LAUNCH:
    server game_port must match between all join requests to client (represents the host's port to sync game clients)
    server base_port
    client game_port must be unique between each client (represents ???)
    client base_port
    client shared_port must match between all join requests to client
        """
        raw,score,feature,rendered = self.interfaces
        interface = sc_pb.InterfaceOptions()
        interface.raw   = raw   # whether raw data is reported in observations
        interface.score = score # whether score data is reported in observations
        interface.feature_layer.width = 24
        #interface.feature_layer.resolution =
        #interface.feature_layer.minimap_resolution =
        joinReq = sc_pb.RequestJoinGame(
            options = interface,
            #observed_player_id=__?__,
            race = self.whoAmI().selectedRace.gameValue())
        # TODO -- allow player to be an observer, not just a player w/ race
        if self.host: # always add ports for joining player to connect to defined host
            hostPorts = self.host[1]
            joinReq.server_ports.game_port = hostPorts[0]
            joinReq.server_ports.base_port = hostPorts[1]
            joinReq.shared_port            = hostPorts[2]
            clientPorts = joinReq.client_ports.add()
            clientPorts.game_port = self.ports[0]
            clientPorts.base_port = self.ports[1]
            ret = self.ports[2]
        elif self.isMultiplayer: # always add ports as host of multiple agents/clients
            if len(self.ports) < 5:
                self.ports += [ # get new private client ports for the host
                    portpicker.pick_unused_port(), # game_port
                    portpicker.pick_unused_port(), # base_port
                ]
            joinReq.server_ports.game_port = self.ports[0]
            joinReq.server_ports.base_port = self.ports[1]
            joinReq.shared_port            = self.ports[2]
            clientPorts = joinReq.client_ports.add()
            clientPorts.game_port = self.ports[3] # new private client game port
            clientPorts.base_port = self.ports[4] # new private client base port
        return joinReq # SC2APIProtocol.RequestJoinGame