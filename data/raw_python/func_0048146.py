def on_connect(self, ws):
    """ Todo """
    self.logger.info("onconnect")
    msg = {'op': self.IDENTIFY,
           'd': {'token': self.token,
                 'properties': {'$os': 'lnx',
                                '$browser': 'discord_simple',
                                '$device': 'discord_simple',
                                '$refferer': '',
                                '$reffering_domain': ''},
                 'compress': False,
                 'large_threshold': 250,
                 'v' : 3}}
    ws.send(json.dumps(msg))