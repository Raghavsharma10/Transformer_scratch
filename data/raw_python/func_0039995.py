def searchTag(self,HTAG="#arenaNETmundial"):
        """Set Twitter search or stream criteria for the selection of tweets"""
        search = t.search(q=HTAG,count=100,result_type="recent")
        ss=search[:]
        search = t.search(q=HTAG,count=150,max_id=ss[-1]['id']-1,result_type="recent")
        #search = t.search(q=HTAG,count=150,since_id=ss[-1]['id'],result_type="recent")
        while seach:
            ss+=search[:]
            search = t.search(q=HTAG,count=150,max_id=ss[-1]['id']-1,result_type="recent")