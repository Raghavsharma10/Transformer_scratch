def makeNetwork(self):
        """Makes graph object from .gdf loaded data"""
        if "weight" in self.data_friendships.keys():
            self.G=G=x.DiGraph()
        else:
            self.G=G=x.Graph()
        F=self.data_friends
        for friendn in range(self.n_friends):
            if "posts" in F.keys():
                G.add_node(F["name"][friendn],
                             label=F["label"][friendn],
                             posts=F["posts"][friendn])
            elif "agerank" in F.keys():
                G.add_node(F["name"][friendn],
                             label=F["label"][friendn],
                             gender=F["sex"][friendn],
                             locale=F["locale"][friendn], 
                             agerank=F["agerank"][friendn])
            else:
                G.add_node(F["name"][friendn],
                             label=F["label"][friendn],
                             gender=F["sex"][friendn],
                             locale=F["locale"][friendn])
        F=self.data_friendships
        for friendshipn in range(self.n_friendships):
            if "weight" in F.keys():
                G.add_edge(F["node1"][friendshipn],F["node2"][friendshipn],weight=F["weight"][friendshipn])
            else:
                G.add_edge(F["node1"][friendshipn],F["node2"][friendshipn])