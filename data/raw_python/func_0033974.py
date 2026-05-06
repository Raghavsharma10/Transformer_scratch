def get_object_cat1(con, token, cat, kwargs):
        """
        Constructs the "GET" URL. The functions is used by the get_object method
        First Category of "GET" URL construction. Again calling it first category because more
        complex functions  maybe added later. 
        """
        req_str = "/"+kwargs['id']+"?"               #/id?
        req_str += "access_token="+token             #/id?@acces_token=......
        del kwargs['id']
        
        key = settings.get_object_cat1_param[cat]    #get the param name for the category(single, multiple)
        req_str += "&"+key+"="                       #/id?@acces_token=......key=
        if key in kwargs.keys():
                length = len( kwargs[key] )
                for i in range(length):
                        if i == 0:
                                req_str +=  kwargs[key][i]
                        else:
                                req_str +=  ","+kwargs[key][i]        
        else:
                return "Parameter Error"
        
        res = wiring.send_request("GET", con, req_str, '')
        return res