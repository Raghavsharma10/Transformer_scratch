def mkzip(source_dir, output_filename):
        '''Usage:
            p = r'D:\auto\env\ttest\ins\build\lib\rock4\softtest\support'
            mkzip(os.path.join(p, "appiumroot"),os.path.join(p, "appiumroot.zip"))
            unzip(os.path.join(p, "appiumroot.zip"),os.path.join(p, "appiumroot2"))  
        '''
        zipf = zipfile.ZipFile(output_filename, 'w', zipfile.zlib.DEFLATED)
        pre_len = len(os.path.dirname(source_dir))
        for parent, dirnames, filenames in os.walk(source_dir):
            for filename in filenames:
                pathfile = os.path.join(parent, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep);#相对路径
                zipf.write(pathfile, arcname)
        zipf.close()