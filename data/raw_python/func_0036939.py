def sub_via_value(self,*vs,**kwargs):
        '''
            d= {1:'a',2:'b',3:'a',4:'d',5:'e'}
            ed = eded.Edict(d)
            ed.sub_via_value('a','d')
        '''
        sd = _select_norecur_via_value(self.dict,*vs,**kwargs)
        return(Edict(sd))