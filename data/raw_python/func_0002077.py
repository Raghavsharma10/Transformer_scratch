def delete(self, id ):
        """ Delete a token """
        target_url = self.client.get_url('TOKEN', 'DELETE', 'single', {'id':id})

        r = self.client.request('DELETE', target_url, headers={'Content-type': 'application/json'})
        r.raise_for_status()