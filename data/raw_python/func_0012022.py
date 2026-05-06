def DyJsonData(cls,name, sequence):
        ''' set dynamic value from the json data of response
        @note: 获取innerHTML json的数据 如，   <html><body>{  "code": 1,"desc": "成功"}</body></html>
        @param name: glob parameter name
        @param sequence: sequence for the json
            e.g.
            result={"a":1,
               "b":[1,2,3,4],
               "c":{"d":5,"e":6},
               "f":{"g":[7,8,9]},
               "h":[{"i":10,"j":11},{"k":12}]
               }
            
            sequence1 ="a" # -> 1
            sequence2 ="b.3" # -> 4
            sequence3 = "f.g.2" # -> 9
            sequence4 = "h.0.j" # -> 11
        '''
        
        cls.SetControl(by = "tag name", value = "body")        
        json_body  = cls._element().get_attribute('innerHTML')
                
        if not json_body:
            return
                
        resp = json.loads(json_body)                    
        sequence = [_parse_string_value(i) for i in sequence.split('.')]    
        for i in sequence:
            try:
                if isinstance(i, int):
                    resp = resp[i]   
                else:
                    resp = resp.get(i)
            except:            
                cls.glob.update({name:None})
                return        
        cls.glob.update({name:resp})