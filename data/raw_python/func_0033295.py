def get(self, uri):
        """Get request
        
        Args:
            uri (str): URI
            
        Returns:
            Json: API response
            
        Raises:
            Exception: Network issue
        """
        r = None
        
        try:
            r = requests.get(uri,
                             allow_redirects=True,
                             timeout=Settings.requests_timeout,
                             headers={},
                             auth=HTTPDigestAuth(self.user, self.password))
            return self.answer(r.status_code, r.json())
        except:
            raise
        finally:
            if r:
                r.connection.close()