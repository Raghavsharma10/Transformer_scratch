def get_switch_actors(self):
        """
        Get information about all actors

        This needs 1+(5n) requests where n = number of actors registered

        Deprecated, use get_actors instead.

        Returns a dict:
        [ain] = {
            'name': Name of actor,
            'state': Powerstate (boolean)
            'present': Connected to server? (boolean)
            'power': Current power consumption in mW
            'energy': Used energy in Wh since last energy reset
            'temperature': Current environment temperature in celsius
        }
        """
        actors = {}
        for ain in self.homeautoswitch("getswitchlist").split(','):
            actors[ain] = {
                'name': self.homeautoswitch("getswitchname", ain),
                'state': bool(self.homeautoswitch("getswitchstate", ain)),
                'present': bool(self.homeautoswitch("getswitchpresent", ain)),
                'power': self.homeautoswitch("getswitchpower", ain),
                'energy': self.homeautoswitch("getswitchenergy", ain),
                'temperature': self.homeautoswitch("getswitchtemperature", ain),
            }
        return actors