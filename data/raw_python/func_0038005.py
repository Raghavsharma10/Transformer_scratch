def _get_ipmitool_path(self, cmd='ipmitool'):
        """Get full path to the ipmitool command using the unix
        `which` command
        """
        p = subprocess.Popen(["which", cmd], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)                
        out, err =  p.communicate()
        return out.strip()