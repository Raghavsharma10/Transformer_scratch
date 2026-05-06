def build_extension(self, ext):
        """
        build clrmagic.dll using csc or mcs
        """
        if sys.platform == "win32":
            _clr_compiler = "C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319\\csc.exe"
        else:
            _clr_compiler = "mcs"
        cmd = [ 
            _clr_compiler,
            "/target:library",
            "clrmagic.cs"
        ]
        check_call(" ".join(cmd), shell=True)