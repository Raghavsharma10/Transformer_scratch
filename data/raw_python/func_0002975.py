def fetch_single_representation(self, item_xlink_href):
        """
        This function will render a formatted URL for accessing the PLoS' server
        SingleRepresentation of an object.
        """
        #A dict of URLs for PLoS subjournals
        journal_urls = {'pgen': 'http://www.plosgenetics.org/article/{0}',
                        'pcbi': 'http://www.ploscompbiol.org/article/{0}',
                        'ppat': 'http://www.plospathogens.org/article/{0}',
                        'pntd': 'http://www.plosntds.org/article/{0}',
                        'pmed': 'http://www.plosmedicine.org/article/{0}',
                        'pbio': 'http://www.plosbiology.org/article/{0}',
                        'pone': 'http://www.plosone.org/article/{0}',
                        'pctr': 'http://clinicaltrials.ploshubs.org/article/{0}'}
        #Identify subjournal name for base URl
        subjournal_name = self.article.doi.split('.')[2]
        base_url = journal_urls[subjournal_name]
        #Compose the address for fetchSingleRepresentation
        resource = 'fetchSingleRepresentation.action?uri=' + item_xlink_href
        return base_url.format(resource)