def getlanguages(self, event):
        """Compile and return a human readable list of registered translations"""

        self.log('Client requests all languages.', lvl=verbose)
        result = {
            'component': 'hfos.ui.clientmanager',
            'action': 'getlanguages',
            'data': language_token_to_name(all_languages())
        }
        self.fireEvent(send(event.client.uuid, result))