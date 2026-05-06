def ret_dump(self, d_ret, **kwargs):
        """
        JSON print results to console (or caller)
        """
        b_print     = True
        for k, v in kwargs.items():
            if k == 'JSONprint':    b_print     = bool(v)
        if b_print:
            print(
                json.dumps(   
                    d_ret, 
                    indent      = 4,
                    sort_keys   = True
                )
        )