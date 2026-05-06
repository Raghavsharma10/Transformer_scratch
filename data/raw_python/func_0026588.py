def _check_flood_protection(self, component, action, clientuuid):
        """Checks if any clients have been flooding the node"""

        if clientuuid not in self._flood_counter:
            self._flood_counter[clientuuid] = 0

        self._flood_counter[clientuuid] += 1

        if self._flood_counter[clientuuid] > 100:
            packet = {
                'component': 'hfos.ui.clientmanager',
                'action': 'Flooding',
                'data': True
            }
            self.fireEvent(send(clientuuid, packet))
            self.log('Flooding from', clientuuid)
            return True