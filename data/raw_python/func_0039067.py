def get_industry_index(self, index_id,items=None):
        """retrieves all symbols that belong to an industry.
        """
        response = self.select('yahoo.finance.industry',items).where(['id','=',index_id])
        return response