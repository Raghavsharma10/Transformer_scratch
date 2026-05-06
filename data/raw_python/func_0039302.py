def removePaging(self,):
        """Remove paging from Binder
        """
        root = self.etree
        t_paging = root.find('paging')

        try:
            root.remove(t_paging)
            return True
        except (Exception,) as e:
            print(e)

        return False