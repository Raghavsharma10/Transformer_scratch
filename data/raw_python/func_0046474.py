def add_docker_file_to_tarfile(self, docker_file, tar):
        """Add a Dockerfile to a tarfile"""
        with hp.a_temp_file() as dockerfile:
            log.debug("Context: ./Dockerfile")
            dockerfile.write("\n".join(docker_file.docker_lines).encode('utf-8'))
            dockerfile.seek(0)
            tar.add(dockerfile.name, arcname="./Dockerfile")