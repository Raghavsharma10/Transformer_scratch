def generate_mobi_file(self):
        '''
        使用 :mod:`subprocess` 模块调用 ``KindleGen`` 工具，
        将已准备好的书籍源文件编译生成 ``mobi`` 文件
        '''
        opf_file = os.path.join(self.build_source_dir, 'moear.opf')
        command_list = [self.kg, opf_file]
        output = subprocess.Popen(
            command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False).communicate()
        self._logger.info('生成命令: {}'.format(' '.join(command_list)))
        self._logger.info('生成 mobi : {}'.format(
            output[0].decode()))
        if output[1]:
            self._logger.error(output[1].decode())
            raise IOError('KindleGen转换失败: {}'.format(output[1]))