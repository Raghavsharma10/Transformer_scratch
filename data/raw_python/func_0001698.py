def echo_percent(self,transferred=1, status=None):
        '''Sample usage:
            f=lambda x,y:x+y
            ldata = range(10)
            toBeTransferred = reduce(f,range(10))
            
            import time
            progress = ProgressBarUtils("viewbar", toBeTransferred=toBeTransferred, run_status="正在下载", fin_status="下载完成")    
            for i in ldata:  
                time.sleep(0.1)  
                progress.echo_percent(i)
        '''
        self.transferred += transferred
        self.status = status or self.status
        end_str = "\r"
        if self.transferred == self.toBeTransferred:
            end_str = '\n'
            self.status = status or self.fin_status
        print(self.__get_bar() + end_str)