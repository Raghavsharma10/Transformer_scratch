def p0(self, e):
        """Returns top level w:p node"""
        body = self.body(e)
        p = self.p(e)
        if body == p.getparent():
            return p
        else:
            raise ElementNotFound("could not find top level w:p element in %s" % e)