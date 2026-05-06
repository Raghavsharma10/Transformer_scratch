def fetch(self, pageNum, itemsPerPage):
        """Intermediate fetching
        
        Args:
            pageNum (int): Page number
            itemsPerPage (int): Number of Users per Page
            
        Returns:
            dict: Response payload
        """
        return self.get_all_alerts(self.status, pageNum, itemsPerPage)