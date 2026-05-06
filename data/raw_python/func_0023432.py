def source_changed(self, event):
        """Generate a simplified chain by joining adjacent transforms.
        """
        # bail out early if the chain is empty
        transforms = self._chain.transforms[:]
        if len(transforms) == 0:
            self.transforms = []
            return
        
        # If the change signal comes from a transform that already appears in
        # our simplified transform list, then there is no need to re-simplify.
        if event is not None:
            for source in event.sources[::-1]:
                if source in self.transforms:
                    self.update(event)
                    return
        
        # First flatten the chain by expanding all nested chains
        new_chain = []
        while len(transforms) > 0:
            tr = transforms.pop(0)
            if isinstance(tr, ChainTransform) and not tr.dynamic:
                transforms = tr.transforms[:] + transforms
            else:
                new_chain.append(tr)
        
        # Now combine together all compatible adjacent transforms
        cont = True
        tr = new_chain
        while cont:
            new_tr = [tr[0]]
            cont = False
            for t2 in tr[1:]:
                t1 = new_tr[-1]
                pr = t1 * t2
                if (not t1.dynamic and not t2.dynamic and not 
                   isinstance(pr, ChainTransform)):
                    cont = True
                    new_tr.pop()
                    new_tr.append(pr)
                else:
                    new_tr.append(t2)
            tr = new_tr

        self.transforms = tr