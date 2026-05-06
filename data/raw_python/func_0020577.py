def totals(self):
        """
        Computes and returns dictionary containing home/away by player, shots and face-off totals
        
        :returns: dict of the form ``{ 'home/away': { 'all_keys': w_numeric_data } }``
        """
        def agg(d):
            keys = ['g','a','p','pm','pn','pim','s','ab','ms','ht','gv','tk','bs']
            res = { k: 0 for k in keys }
            res['fo'] = { 'won': 0, 'total': 0 }
            for _, v in d.items():
                for k in keys:
                    res[k] += v[k]
                for fi in res['fo'].keys():
                    res['fo'][fi] += v['fo'][fi]
            return res
            
        return self.__apply_to_both(agg)