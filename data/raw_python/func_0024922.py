def _create_connection(self):
        """
        Create a new websocket connection with proper headers.
        """
        logging.debug("Initializing new websocket connection.")
        headers = {
            'Authorization': self.service._get_bearer_token(),
            'Predix-Zone-Id': self.ingest_zone_id,
            'Content-Type': 'application/json',
        }
        url = self.ingest_uri

        logging.debug("URL=" + str(url))
        logging.debug("HEADERS=" + str(headers))

        # Should consider connection pooling and longer timeouts
        return websocket.create_connection(url, header=headers)