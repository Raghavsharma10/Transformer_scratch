def new_backup(self, src):
        """ Create a new backup file allocation """

        backup_id_file = p.join(self.backup_dir, '.bk_idx')
        backup_num = file_or_default(backup_id_file, 1, int)
        backup_name = str(backup_num) + "_" + os.path.basename(src)
        backup_num += 1

        file_put_contents(backup_id_file, str(backup_num))
        return p.join(self.backup_dir, backup_name)