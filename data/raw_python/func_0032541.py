def rows(self):
        """Returns a numpy array of the rows name"""
        bf = self.copy()
        result = bf.query.executeQuery(format="soa")
        return result["_rowName"]