def UploadType(cls, file_path):
        """    上传，  一般，上传页面如果是input,原生file文件框, 如： <input type="file" id="test-image-file" name="test" accept="image/gif">，像这样的，定位到该元素，然后使用 send_keys 上传的文件的绝对路径        
        @param file_name: 文件名(文件必须存在在工程resource目录下)
        """
        if not os.path.isabs(file_path):
            return False
        
        if os.path.isfile(file_path):
            cls.SendKeys(file_path)
        else:            
            return False