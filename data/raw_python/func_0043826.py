def updateIDs(self, ginfo, tag=None, debug=False):
        """ensure all player's playerIDs are correct given game's info"""
            # SC2APIProtocol.ResponseGameInfo attributes:
                # map_name
                # mod_names
                # local_map_path
                # player_info
                # start_raw
                # options
        thisPlayer = self.whoAmI()
        for pInfo in ginfo.player_info: # parse ResponseGameInfo.player_info to validate player information (SC2APIProtocol.PlayerInfo) against the specified configuration
            pID = pInfo.player_id
            if pID == thisPlayer.playerID: continue # already updated
            pCon = c.types.PlayerControls(pInfo.type)
            rReq = c.types.SelectRaces(pInfo.race_requested)
            for p in self.players: # ensure joined player is identified appropriately
                if p.playerID and p.playerID != pID: continue # if this non-matching player already has a set playerID, it can't match
                if p.control == pCon and p.selectedRace == rReq: # matched player
                    p.playerID = pID # updated player IDs should be saved into the game configuration
                    if debug: print("[%s] match contains %s."%(tag, p))
                    pID = 0 # declare that the player has been identified
                    break
            if pID: raise c.UnknownPlayer("could not match %s %s %s to any "
                "existing player of %s"%(pID, pCon, rReq, self.players))