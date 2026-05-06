def add_info(self, data):
        """add info to a build"""
        for key in data:
            # verboten
            if key in ('status','state','name','id','application','services','release'):
                raise ValueError("Sorry, cannot set build info with key of {}".format(key))
            self.obj[key] = data[key]
        self.changes.append("Adding build info")
        return self