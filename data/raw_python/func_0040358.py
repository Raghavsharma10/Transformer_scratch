def searchTag(self,HTAG="#python"):
        """Set Twitter search or stream criteria for the selection of tweets"""
        self.t = Twython(app_key           =self.app_key           ,
                        app_secret         =self.app_secret        ,
                        oauth_token        =self.oauth_token       ,
                        oauth_token_secret =self.oauth_token_secret)

        search =self.t.search(q=HTAG,count=100,result_type="recent")
        ss=search[:]
        search = self.t.search(q=HTAG,count=150,max_id=ss[-1]['id']-1,result_type="recent")
        #search = t.search(q=HTAG,count=150,since_id=ss[-1]['id'],result_type="recent")
        while seach:
            ss+=search[:]
            search = self.t.search(q=HTAG,count=150,max_id=ss[-1]['id']-1,result_type="recent")
        self.ss=ss