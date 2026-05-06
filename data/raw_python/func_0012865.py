def getrefs(self, reflist):
        """
        reflist is got from getobjectref in parse_idd.py
        getobjectref returns a dictionary.
        reflist is an item in the dictionary
        getrefs gathers all the fields refered by reflist
        """
        alist = []
        for element in reflist:
            if element[0].upper() in self.dt:
                for elm in self.dt[element[0].upper()]:
                    alist.append(elm[element[1]])
        return alist