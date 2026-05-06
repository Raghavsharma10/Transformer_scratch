def addPaging(self,paging):
        """Add paging to Binder
        """
        if not vars(self).get('paging', None):
            self.paging = paging
        root = self.etree

        try:
            root.append(paging.etree)
            return True
        except (Exception,) as e:
            print(e)

        return False