def get_3_tuple(self,obj,default=None):
        """Return 3-tuple from
        number -> (obj,default[1],default[2])
        0-sequence|None -> default
        1-sequence -> (obj[0],default[1],default[2])
        2-sequence -> (obj[0],obj[1],default[2])
        (3 or more)-sequence -> (obj[0],obj[1],obj[2])
        """
        if not (default is not None \
                and type(default) is tuple \
                and len(default)==3):
            raise ValueError('argument default must be 3-tuple|None but got %s'%(default))
        if is_sequence(obj):
            n = len(obj)
            if n>3:
                log.warning('expected 3-sequence but got %s-%s'%(n,type(obj)))
            if n>=3:
                return tuple(obj)
            log.warning('filling with default value (%s) to obtain size=3'%(default[0]))
            if default is not None:
                if n==0:
                    return default
                elif n==1:
                    return (obj[0],default[1],default[2])
                elif n==2:
                    return (obj[0],obj[1],default[2])
        elif is_number(obj) and default is not None:
            log.warning('filling with default value (%s) to obtain size=3'%(default[0]))
            return (obj,default[1],default[2])
        elif obj is None and default is not None:
            log.warning('filling with default value (%s) to obtain size=3'%(default[0]))
            return default
        raise ValueError('failed to construct 3-tuple from %s-%s'%(n,type(obj)))