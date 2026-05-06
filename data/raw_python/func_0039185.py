def compare_registry(self, concurrent=False):
        """Compares the Windows Registry contained within the two File Systems.

        It parses all the registry hive files contained within the disks
        and generates the following report.

            {'created_keys': {'\\Reg\\Key': (('Key', 'Type', 'Value'))}
             'deleted_keys': ['\\Reg\\Key', ...],
             'created_values': {'\\Reg\\Key': (('Key', 'Type', 'NewValue'))},
             'deleted_values': {'\\Reg\\Key': (('Key', 'Type', 'OldValue'))},
             'modified_values': {'\\Reg\\Key': (('Key', 'Type', 'NewValue'))}}

        Only registry hives which are contained in both disks are compared.
        If the second disk contains a new registry hive,
        its content can be listed using winreg.RegistryHive.registry() method.

        If the concurrent flag is True,
        two processes will be used speeding up the comparison on multiple CPUs.

        """
        self.logger.debug("Comparing Windows registries.")

        self._assert_windows()

        return compare_registries(self.filesystems[0], self.filesystems[1],
                                  concurrent=concurrent)