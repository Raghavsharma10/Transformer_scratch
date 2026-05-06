def automatic_syn_data_from_mongo(self,
                                      index,
                                      doc_type,
                                      indexed_flag_field_name,
                                      thread_name='automatic_syn_data_thread',
                                      interval=60,
                                      use_mongo_id=False,
                                      mongo_query_params={},
                                      mongo_host=default.MONGO_HOST,
                                      mongo_port=default.MONGO_PORT,
                                      mongo_db=default.MONGO_DB,
                                      mongo_collection=default.MONGO_COLLECTION):
        """
        Automatic synchronize data that from MongoDB into the Elasticsearch by schedule task,
        it will synchronize this data if the indexed_flag_field_name of the field of the document is False.
        Noteworthy that the function may be no good please you caution use it.

        :param indexed_flag_field_name: the name of the field of the document,
                    if associated value is False will synchronize data for it
        :param thread_name: the name of the schedule task thread
        :param interval: the time that executes interval of the scheduled task every time (unit second)
        :return: the thread id, you can use this id to cancel associated task
        """
        thread_id = self._generate_thread_id(thread_name)
        if thread_id in ElasticsearchClient.automatic_syn_data_flag:
            lock.acquire()
            try:
                thread_name = thread_name + '-%s' % ElasticsearchClient.automatic_thread_name_counter
                ElasticsearchClient.automatic_thread_name_counter += 1
                thread_id = self._generate_thread_id(thread_name)
            finally:
                lock.release()
        ElasticsearchClient.automatic_syn_data_flag[thread_id] = True

        t = threading.Thread(target=ElasticsearchClient._automatic_syn_data_from_mongo_worker,
                             args=(self, thread_id, index, doc_type,
                                   indexed_flag_field_name, interval, use_mongo_id,
                                   mongo_query_params,
                                   mongo_host, mongo_port,
                                   mongo_db, mongo_collection),
                             name=thread_name)

        t.start()
        return thread_id