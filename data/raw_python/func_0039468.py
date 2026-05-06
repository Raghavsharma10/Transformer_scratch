def rulefor(self, addr):
        """Return the rule object for an address from our deps graph."""
        return self.rule.subgraph.node[self.rule.makeaddress(addr)][
            'target_obj']