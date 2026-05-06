def handle(self, *args, **options):
        """Run do_index_command on each specified index and log the output."""
        for index in options.pop("indexes"):
            data = {}
            try:
                data = self.do_index_command(index, **options)
            except TransportError as ex:
                logger.warning("ElasticSearch threw an error: %s", ex)
                data = {"index": index, "status": ex.status_code, "reason": ex.error}
            finally:
                logger.info(data)