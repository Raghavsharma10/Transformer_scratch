def init_project_env(subject='Automation', proj_path = None, sysencoding = "utf-8", debug = False):
    ''' Set the environment for pyrunner '''    
        
#     if sysencoding:
#         set_sys_encode(sysencoding)
    
    if not proj_path:
        try:
            executable_file_path = os.path.dirname(os.path.abspath(inspect.stack()[-1][1]))
        except:
            executable_file_path = os.path.dirname(sys.path[0])
        finally:
            proj_path = executable_file_path
    
    p = os.path.join(proj_path,subject)
    
    proj_conf = {
        "sys_coding" : sysencoding,
        "debug" : debug,
        "module_name" : os.path.splitext(os.path.basename(subject))[0],
        "cfg_file" : os.path.join(p,"config.ini"),
        "path" : {"root" : p,
                  "case" : os.path.join(p,"testcase"),
                  "data" : os.path.join(p,"data"),
                  "buffer" : os.path.join(p,"buffer"),
                  "resource" : os.path.join(p,"resource"),
                  "tools" : os.path.join(p,"tools"),
                  "rst" : os.path.join(p,"result"),
                  "rst_log" : os.path.join(p,"result","testcase"),
                  "rst_shot" : os.path.join(p,"result","screenshots"),
            },
        }
     
    [FileSystemUtils.mkdirs(v) for v in proj_conf["path"].values()]    
    sys.path.append(p) if os.path.isdir(p) else ""
    return proj_conf