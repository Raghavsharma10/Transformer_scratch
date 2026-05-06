def clone_with_new_dockerfile(self, conf, docker_file):
        """Clone this tarfile and add in another filename before closing the new tar and returning"""
        log.info("Copying context to add a different dockerfile")
        self.close()
        with a_temp_file() as tmpfile:
            old_t = os.stat(self.tmpfile.name).st_size > 0
            if old_t:
                shutil.copy(self.tmpfile.name, tmpfile.name)

            with tarfile.open(tmpfile.name, mode="a") as t:
                conf.add_docker_file_to_tarfile(docker_file, t)
                yield ContextWrapper(t, tmpfile)