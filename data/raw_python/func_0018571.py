def send_buffered_operations(self):
        """Send buffered operations to Elasticsearch.

        This method is periodically called by the AutoCommitThread.
        """
        with self.lock:
            try:
                action_buffer = self.BulkBuffer.get_buffer()
                if action_buffer:
                    successes, errors = bulk(self.elastic, action_buffer)
                    LOG.debug("Bulk request finished, successfully sent %d "
                              "operations", successes)
                    if errors:
                        LOG.error(
                            "Bulk request finished with errors: %r", errors)
            except es_exceptions.ElasticsearchException:
                LOG.exception("Bulk request failed with exception")