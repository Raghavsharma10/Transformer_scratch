def getInferredPropertiesForClass(self, aClass, rel="domain_of"):
        """
        returns all properties valid for a class (as they have it in their domain)
        recursively ie traveling up the descendants tree
        Note: results in a list of dicts including itself
        Note [2]: all properties with no domain info are added at the top as [None, props]

        :return:
        [{<Class *http://xmlns.com/foaf/0.1/Person*>:
            [<Property *http://xmlns.com/foaf/0.1/currentProject*>,<Property *http://xmlns.com/foaf/0.1/familyName*>,
                etc....]},
        {<Class *http://www.w3.org/2003/01/geo/wgs84_pos#SpatialThing*>:
            [<Property *http://xmlns.com/foaf/0.1/based_near*>, etc...]},
            ]
        """
        _list = []

        if rel=="domain_of":
            _list.append({aClass: aClass.domain_of})
            for x in aClass.ancestors():
                if x.domain_of:
                    _list.append({x: x.domain_of})

            # add properties from Owl:Thing ie the inference layer

            topLevelProps = [p for p in self.properties if p.domains == []]
            if topLevelProps:
                _list.append({self.OWLTHING: topLevelProps})

        elif rel=="range_of":
            _list.append({aClass: aClass.range_of})
            for x in aClass.ancestors():
                if x.domain_of:
                    _list.append({x: x.range_of})

            # add properties from Owl:Thing ie the inference layer

            topLevelProps = [p for p in self.properties if p.ranges == []]
            if topLevelProps:
                _list.append({self.OWLTHING: topLevelProps})

        return _list