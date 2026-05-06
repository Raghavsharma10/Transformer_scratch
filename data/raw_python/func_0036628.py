def search_image(self, search_term):
        """
        Search for a specific image by providing a search term (mainly used with ec2's community and public images)

        :param search_term: Search term to be used when searching for images's names containing this term.
        :returns: A list of all images, whose names contain the given search_term.
        """
        payload = {
            'search_term': search_term
        }
        data = json.dumps(payload)

        req = self.request(self.mist_client.uri+'/clouds/'+self.id+'/images', data=data)
        images = req.get().json()
        return images