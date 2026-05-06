def debug(self, *debugReqs):
        """send a debug command to control the game state's setup"""
        return self._client.send(debug=sc2api_pb2.RequestDebug(debug=debugReqs))