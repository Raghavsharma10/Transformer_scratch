def execute(self, requete_SQL):
        """Execute one or many requests
        requete_SQL may be a tuple(requete,args) or a list of such tuples
        Return the result or a list of results
        """
        try:
            cursor = self.cursor()
            if isinstance(requete_SQL,tuple):
                res = self._execute_one(cursor,*requete_SQL)
            else:
                res = []
                for r in requete_SQL:
                    if r:
                        res.append(self._execute_one(cursor,*r))

        except self.SQL.Error as e:
            raise StructureError(f"SQL error ! Details : \n {e}")
        else:
            self.connexion.commit()
        finally:
            self.connexion.close()
        return res