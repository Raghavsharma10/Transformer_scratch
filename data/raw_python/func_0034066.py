def delete(self,  **kwargs):       
                """
                Used for deleting objects from the facebook graph. Just pass the id of the object to be 
                deleted. But in case of like, have to pass the cat ("likes") and object id as a like has no id
                itself in the facebook graph
                """
                if 'cat' not in kwargs.keys():
                        kwargs['cat']=''
                cat=kwargs['cat']
                del kwargs['cat']
                res=request.publish_cat1("DELETE", self.con, self.token,  cat, kwargs)
                return res