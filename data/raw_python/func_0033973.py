def publish_cat1(method, con, token, cat, kwargs):
        """
        Constructs a "POST" and "DELETE" URL. The function is used by the publish and delete method
        First category of  "POST" and "DELETE" url construction. Caling it first category because for 
        publishing photos or more complex stuffs, newer fucntions might be added to deal with "POST". 
        """
        req_str = "/"+str( kwargs['id'] )+"/"+cat+'?'                #/id/category?
        del kwargs['id']
        kwargs['access_token'] = token                               #add access token to kwwargs
        res = wiring.send_request(method, con, req_str, kwargs)    
        return res