def head(self, n=5):
        """Returns first n rows"""
        col = self.copy()
        col.query.setLIMIT(n)
        return col.toPandas()