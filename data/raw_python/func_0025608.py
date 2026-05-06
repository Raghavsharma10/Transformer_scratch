def execute_command_in_dir(command, directory, verbose=DEFAULTS['v'], 
                           prefix="Output: ", env=None):
    
    """Execute a command in specific working directory"""
    
    if os.name == 'nt':
        directory = os.path.normpath(directory)
        
    print_comment("Executing: (%s) in directory: %s" % (command, directory),
                  verbose)
    if env is not None:
        print_comment("Extra env variables %s" % (env), verbose)
    
    try:
        if os.name == 'nt':
            return_string = subprocess.check_output(command, 
                                                    cwd=directory, 
                                                    shell=True, 
                                                    env=env,
                                                    close_fds=False)
        else:
            return_string = subprocess.check_output(command, 
                                                    cwd=directory, 
                                                    shell=True, 
                                                    stderr=subprocess.STDOUT,
                                                    env=env,
                                                    close_fds=True)
        
        return_string = return_string.decode("utf-8") # For Python 3
                                
        print_comment('Command completed. Output: \n %s%s' % \
                      (prefix,return_string.replace('\n','\n '+prefix)), 
                      verbose)

        return return_string
    
    except AttributeError:
        # For python 2.6...
        print_comment_v('Assuming Python 2.6...')
        
        return_string = subprocess.Popen(command, 
                                         cwd=directory, 
                                         shell=True,
                                         stdout=subprocess.PIPE).communicate()[0]
        return return_string
    
    except subprocess.CalledProcessError as e:        
        
        print_comment_v('*** Problem running command: \n       %s'%e)
        print_comment_v('%s%s'%(prefix,e.output.decode().replace('\n','\n'+prefix)))
        
        return None
        
    except:
        print_comment_v('*** Unknown problem running command: %s'%e)
        
        return None
        
    print_comment("Finished execution", verbose)