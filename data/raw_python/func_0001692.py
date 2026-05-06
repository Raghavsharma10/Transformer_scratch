def find_data_files(source,target,patterns,isiter=False):
        """Locates the specified data-files and returns the matches; 
            filesystem tree for setup's data_files in setup.py
            Usage:
                data_files = find_data_files(r"C:\Python27\Lib\site-packages\numpy\core","numpy/core",["*.dll","*.pyd"])
                data_files = find_data_files(r"d:\auto\buffer\test\test","buffer/test/test",["*"],True)
            :param source -a full path directory which you want to find data from
            :param target -a relative path directory which you want to pack data to
            :param patterns -glob patterns, such as "*dll", "*pyd"  etc.
            :param isiter - True/Fase, Will traverse path if True when patterns equal ["*"] 
        """
        if glob.has_magic(source) or glob.has_magic(target):
            raise ValueError("Magic not allowed in src, target")
        ret = {}
        for pattern in patterns:
            pattern = os.path.join(source,pattern)
            for filename in glob.glob(pattern):
                if os.path.isfile(filename):
                    targetpath = os.path.join(target,os.path.relpath(filename,source))
                    path = os.path.dirname(targetpath)
                    ret.setdefault(path,[]).append(filename)
                elif isiter and os.path.isdir(filename):
                    source2 = os.path.join(source,filename)
                    targetpath2 = "%s/%s" %(target,os.path.basename(filename))
                    # iter_target = os.path.dirname(targetpath2)
                    ret.update(SetupUtils.find_data_files(source2,targetpath2,patterns,isiter))
                 
        return sorted(ret.items())