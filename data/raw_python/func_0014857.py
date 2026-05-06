def get_3_3_tuple(self,obj,default=None):
        """Return tuple of 3-tuples
        """
        if is_sequence2(obj):
            ret = []
            for i in range(3):
                if i<len(obj):
                    ret.append(self.get_3_tuple(obj[i],default))
                else:
                    ret.append(self.get_3_tuple(default,default))
            return tuple(ret)
        if is_sequence(obj):
            if len(obj)>9:
                log.warning('ignoring elements obj[i], i>=9')
            r = obj[:9]
            r = [self.get_3_tuple(r[j:j+3],default) for j in range(0,len(r),3)]
            if len(r)<3:
                log.warning('filling with default value (%s) to obtain size=3'%(default[0]))
            while len(r)<3:
                r.append(self.get_3_tuple(default,default))
            return tuple(r)
        log.warning('filling with default value (%s) to obtain size=3'%(default[0]))
        r1 = self.get_3_tuple(obj,default)
        r2 = self.get_3_tuple(default,default)
        r3 = self.get_3_tuple(default,default)
        return (r1,r2,r3)