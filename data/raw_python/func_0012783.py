def idfstr(self):
        """String representation of the IDF.

        Returns
        -------
        str

        """
        if self.outputtype == 'standard':
            astr = ''
        else:
            astr = self.model.__repr__()

        if self.outputtype == 'standard':
            astr = ''
            dtls = self.model.dtls
            for objname in dtls:
                for obj in self.idfobjects[objname]:
                    astr = astr + obj.__repr__()
        elif self.outputtype == 'nocomment':
            return astr
        elif self.outputtype == 'nocomment1':
            slist = astr.split('\n')
            slist = [item.strip() for item in slist]
            astr = '\n'.join(slist)
        elif self.outputtype == 'nocomment2':
            slist = astr.split('\n')
            slist = [item.strip() for item in slist]
            slist = [item for item in slist if item != '']
            astr = '\n'.join(slist)
        elif self.outputtype == 'compressed':
            slist = astr.split('\n')
            slist = [item.strip() for item in slist]
            astr = ' '.join(slist)
        else:
            raise ValueError("%s is not a valid outputtype" % self.outputtype)
        return astr