def url(self):
        '''
        Executes the methods to send request, process the response and then
        publishes the url.
        '''
        self.get_response()
        url = self.process_response()

        if url:
            logging.info('Your paste has been published at %s' %(url))
            return url
        else:
            logging.error('Did not get a URL back for the paste')
            raise PasteException("No URL for paste")