def _process_list(self, list_line):
        # -rw-r--r-- 1 ftp ftp 68	 May 09 19:37 testftp.txt
        """
            Processes a line of 'ls -l' output, and updates state accordingly.

        :param list_line: Line to process
        """
        res = list_line.split(' ', 8)
        if res[0].startswith('-'):
            self.state['file_list'].append(res[-1])
        if res[0].startswith('d'):
            self.state['dir_list'].append(res[-1])