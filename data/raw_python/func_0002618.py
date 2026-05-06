def _rank_init(self,unranked):
        """Computes rank of provided unranked list of vertices and all
           their children. A vertex will be asign a rank when all its 
           inward edges have been *scanned*. When a vertex is asigned
           a rank, its outward edges are marked *scanned*.
        """
        assert self.dag
        scan = {}
        # set rank of unranked based on its in-edges vertices ranks:
        while len(unranked)>0:
            l = []
            for v in unranked:
                self.setrank(v)
                # mark out-edges has scan-able:
                for e in v.e_out(): scan[e]=True
                # check if out-vertices are rank-able:
                for x in v.N(+1):
                    if not (False in [scan.get(e,False) for e in x.e_in()]):
                        if x not in l: l.append(x)
            unranked=l