def save_to_json(self):
        """The method saves data to json from object"""

        requestvalues = {
            'id': self.dataset,
            'publicationDate': self.publication_date.strftime('%Y-%m-%d'),
            'source': self.source,
            'refUrl': self.refernce_url,
        }        
        return json.dumps(requestvalues)