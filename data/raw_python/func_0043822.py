def requestCreateDetails(self):
        """add configuration to the SC2 protocol create request"""
        createReq = sc_pb.RequestCreateGame( # used to advance to Status.initGame state, when hosting
            realtime    = self.realtime,
            disable_fog = self.fogDisabled,
            random_seed = int(time.time()), # a game is created using the current second timestamp as the seed
            local_map   = sc_pb.LocalMap(map_path=self.mapLocalPath,
                                         map_data=self.mapData))
        for player in self.players:
            reqPlayer = createReq.player_setup.add() # add new player; get link to settings
            playerObj = PlayerPreGame(player)
            if playerObj.isComputer:
                reqPlayer.difficulty    = playerObj.difficulty.gameValue()
            reqPlayer.type              = c.types.PlayerControls(playerObj.control).gameValue()
            reqPlayer.race              = playerObj.selectedRace.gameValue()
        return createReq # SC2APIProtocol.RequestCreateGame