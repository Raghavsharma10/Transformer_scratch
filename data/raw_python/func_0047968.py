def the_context(self, content, silent_build=False):
        """Return either a file with the content written to it, or a whole new context tar"""
        if isinstance(content, six.string_types):
            with a_temp_file() as fle:
                fle.write(content.encode('utf-8'))
                fle.seek(0)
                yield fle
        elif "context" in content:
            with ContextBuilder().make_context(content["context"], silent_build=silent_build) as wrapper:
                wrapper.close()
                yield wrapper.tmpfile
        elif "image" in content:
            from harpoon.ship.runner import Runner
            with a_temp_file() as fle:
                content["conf"].command = "yes"
                with Runner()._run_container(content["conf"], content["images"], detach=True, delete_anyway=True):
                    try:
                        strm, stat = content["docker_api"].get_archive(content["conf"].container_id, content["path"])
                    except docker.errors.NotFound:
                        raise BadOption("Trying to get something from an image that don't exist!", path=content["path"], image=content["conf"].image_name)
                    else:
                        log.debug(stat)

                        fo = BytesIO(b''.join(strm))

                        # In newer docker the archive is a gzipped archive
                        # But in older docker, it's a normal tar
                        for mode in ("r:gz", "r"):
                            try:
                                tf = tarfile.open(fileobj=fo, mode=mode)
                                break
                            except tarfile.ReadError:
                                if mode == "r":
                                    raise
                                fo.seek(0)

                        if tf.firstmember.isdir():
                            tf2 = tarfile.TarFile(fileobj=fle, mode='w')
                            name = tf.firstmember.name
                            for member in tf.getmembers()[1:]:
                                member.name = member.name[len(name)+1:]
                                if member.issym():
                                    with tempfile.NamedTemporaryFile() as symfle:
                                        os.remove(symfle.name)
                                        os.symlink(member.linkpath, symfle.name)
                                        tf2.addfile(member, fileobj=symfle)
                                elif not member.isdir():
                                    tf2.addfile(member, fileobj=tf.extractfile(member.name))
                            tf2.close()
                        else:
                            fle.write(tf.extractfile(tf.firstmember.name).read())

                        tf.close()
                        log.info("Got '{0}' from {1} for context".format(content["path"], content["conf"].container_id))

                fle.seek(0)
                yield fle