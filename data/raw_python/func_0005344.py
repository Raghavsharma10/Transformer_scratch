def find_first_wt_parent(self, with_ip=False):
        """
        Recursively looks at the part_of parent ancestry line (ignoring pooled_from parents) and returns
        a parent Biosample ID if its wild_type attribute is True. 

        Args:
            with_ip: `bool`. True means to restrict the search to the first parental Wild Type that 
                also has an Immunoblot linked to it, which may serve as a control between another 
                immunoblot. For example, it could be useful to compare the target protein bands in
                Immunoblots between a Wild Type sample and a CRISPR eGFP-tagged gene in a 
                descendent sample. 

        Returns:
            `False`: There isn't a WT parent, or there is but not one with an Immunoblot linked to
                it (if the `with_ip` parameter is set to True). 
            `int`: The ID of the WT parent. 
        """
        parent_id = self.part_of_id
        if not parent_id:
            return False
        parent = Biosample(parent_id)
        if parent.wild_type:
            if with_ip and parent.immunoblot_ids:
                return parent.id
            elif not with_ip:
                return parent.id
        return parent.find_first_wt_parent(with_ip=with_ip)