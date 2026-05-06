def handle_next_export(self):
        """
        Retrieve and fully evaluate the next export task, including resolution of any sub-tasks requested by the
        import client such as requests for binary data, observation, etc.

        :return:
            An instance of ExportStateCache, the 'state' field contains the state of the export after running as many
            sub-tasks as required until completion or failure. If there were no jobs to run this returns None.

                :complete:
                    A job was processed and completed. The job has been marked as complete in the database
                :continue:
                    A job was processed, more information was requested and sent, but the job is still active
                :failed:
                    A job was processed but an error occurred during processing
                :confused:
                    A job was processed, but the importer returned a response which we couldn't recognise
        """
        state = None
        while True:
            state = self._handle_next_export_subtask(export_state=state)
            if state is None:
                return None
            elif state.export_task is None:
                return state