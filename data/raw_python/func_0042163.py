def prepare_files(self, target_dir):
        """
        Proper version of file needs to be moved to external directory.
        Because: 1. local files can differ from commited, 2. we can push man branches
        """
        diff_names = self.git_wrapper.get_diff_names(self.remote_sha1, self.local_sha1)
        files_modified = diff_names.split('\n')
        extensions = LINTERS.keys()

        for file_path in files_modified:
            extension = file_path.split('.')[-1]
            if extension not in extensions:
                continue

            new_file_path = os.path.join(target_dir, file_path)
            new_dirname = os.path.dirname(new_file_path)
            if not os.path.isdir(new_dirname):
                os.makedirs(new_dirname)

            with open(new_file_path, "wb") as fh:
                self.git_wrapper.save_content_to_file(file_path, self.local_ref, fh)
            yield new_file_path