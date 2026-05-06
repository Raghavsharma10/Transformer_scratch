def getResults(self, parFound = None):
        ''' 
            Function to obtain the Dictionarythat represents this object.
            
            :param parFound:    values to return.

            :return:    The output format will be like:
                [{"type" : "i3visio.email", "value": "foo@bar.com", "attributes": [] }, {"type" : "i3visio.email", "value": "bar@foo.com", "attributes": [] }]
        '''
        # Defining a dictionary
        results = []
        # Defining a dictionary inside with a couple of fields: reg_exp for the regular expression and found_exp for the expressions found.
        #results[self.name] = {"reg_exp" : self.reg_exp, "found_exp" : parFound}
        #results[self.name] = parFound
        if len(parFound ) >0:
            for found in parFound:
                aux = {}
                aux["type"] = self.name
                aux["value"] = found
                aux["attributes"] = self.getAttributes(found)
                results.append(aux)
        return results