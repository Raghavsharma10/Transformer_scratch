def echo_size(self, transferred=1, status=None):
        '''Sample usage:
            
            f=lambda x,y:x+y
            ldata = range(10)
            toBeTransferred = reduce(f,range(10))
            
            progress = ProgressBarUtils("refresh", toBeTransferred=toBeTransferred, unit="KB", chunk_size=1.0, run_status="正在下载", fin_status="下载完成")
            import time
            for  i in ldata:
                time.sleep(0.2)
                progress.echo_size(transferred=i)
        '''
        self.transferred += transferred
        # if status is not None:
        self.status = status or self.status
        end_str = "\r"
        if self.transferred == self.toBeTransferred:
            end_str = '\n'
            self.status = status or self.fin_status
        
        print(self.__get_info() + end_str)