def publish(self,  cat, **kwargs):
                """
                This method is used for creating objects in the facebook graph.
                The first paramter is "cat", the category of publish. In addition to "cat"
                "id" must also be passed and is catched by "kwargs"
                """
                res=request.publish_cat1("POST", self.con, self.token,  cat, kwargs)    
                return res