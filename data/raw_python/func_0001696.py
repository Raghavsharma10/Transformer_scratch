def echo(transferred, toBeTransferred, suffix=''):
        ''' usage:
            for i in range(101):
                ProgressBarUtils.echo(i,100)
        '''
        bar_len = 60                
        rate = transferred/float(toBeTransferred)
        
        filled_len = int(round(bar_len * rate))
        _percents = "%s%s" %(round(100.0 * rate, 1), "%")
        
        end_str = "\r"
        _bar = '=' * filled_len + '-' * (bar_len - filled_len)
        print("[%s] %s ...%s%s" %(_bar, _percents, suffix, end_str))