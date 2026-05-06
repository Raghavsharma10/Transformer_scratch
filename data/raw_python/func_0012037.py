def Upload(cls, filename):
        """        文件上传， 非原生input
        @todo:  some  upload.exe not  prepared
        @param file: 文件名(文件必须存在在工程resource目录下), upload.exe工具放在工程tools目录下
        """
        raise Exception("to do")
    
        TOOLS_PATH = ""
        RESOURCE_PATH = ""
        tool_4path = os.path.join(TOOLS_PATH, "upload.exe")        
        file_4path = os.path.join(RESOURCE_PATH, filename)
        #file_4path.decode('utf-8').encode('gbk')
        
        if os.path.isfile(file_4path):
            cls.Click()
            os.system(tool_4path + ' ' + file_4path)
        else:
            raise Exception('%s is not exists' % file_4path)