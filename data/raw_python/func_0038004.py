def _expect_method(self, command):
        """Use the expect module to execute ipmitool commands
        and set status
        """
        child = pexpect.spawn(self._ipmitool_path, self.args + command)
        
        i = child.expect([pexpect.TIMEOUT, 'Password: '], timeout=10)
        if i == 0:
            child.terminate()
            self.error = 'ipmitool command timed out'
            self.status = 1
        else:
            child.sendline(self.password)
        
        i = child.expect([pexpect.TIMEOUT, pexpect.EOF], timeout=10)
        if i == 0:
            child.terminate()
            self.error = 'ipmitool command timed out'
            self.status = 1
        else:
            if child.exitstatus:
                self.error = child.before
            else:
                self.output = child.before

            self.status = child.exitstatus
            child.close()