def get_object(self,  cat, **kwargs):
                """
                This method is used for retrieving objects from facebook. "cat", the category, must be
                passed. When cat is "single", pass the "id "and desired "fields" of the single object. If the 
                cat is "multiple", only pass the "ids" of the objects to be fetched.
                """
                if 'id' not in kwargs.keys():
                        kwargs['id']=''
                res=request.get_object_cat1(self.con, self.token, cat,  kwargs)
                return res