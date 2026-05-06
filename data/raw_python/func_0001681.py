def until_cmd(listcmd, end_expects=None, save2logfile=None, coding = encoding):
        ''' 执行系统命令,并等待执行完
            @param listcmd: 执行的命令，列表格式
            @param end_expects: 命令执行结束，在输出的最后一行，正则搜素期望值，并设置 结果标志
            @param save2logfile:  设置执行过程，保存的日志
            @param coding: 设置输出编码        
        '''
        
        
        if end_expects and not isinstance(end_expects, p_compat.str):
            raise Exception("invalide unicode string: '%s'" %end_expects)
        
        lines = []    
        subp = subprocess.Popen(listcmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        while subp.poll()==None:
            next_line = subp.stdout.readline().decode(coding)
            if next_line:
#                 print(next_line)
                lines.append(next_line)
                if end_expects and re.search(end_expects, next_line):
                    result = True
                else:
                    result = False        
        subp.stdout.close()
        
        if subp.returncode:
            result = False
            lines.append("sub command error code: %s" %subp.returncode)
        
        if save2logfile:
            with open(save2logfile, 'a') as f:
                f.writelines(lines)
                                    
        return result