def from_dict(self, d):
        """
        Create a Task from a dictionary. The change is in inplace.

        :argument: python dictionary
        :return: None
        """

        if 'uid' in d:
            if d['uid']:
                self._uid = d['uid']

        if 'name' in d:
            if d['name']:
                self._name = d['name']

        if 'state' in d:
            if isinstance(d['state'], str) or isinstance(d['state'], unicode):
                self._state = d['state']
            else:
                raise TypeError(entity='state', expected_type=str,
                                actual_type=type(d['state']))
        else:
            self._state = states.INITIAL

        if 'state_history' in d:
            if isinstance(d['state_history'], list):
                self._state_history = d['state_history']
            else:
                raise TypeError(entity='state_history', expected_type=list, actual_type=type(
                    d['state_history']))

        if 'pre_exec' in d:
            if isinstance(d['pre_exec'], list):
                self._pre_exec = d['pre_exec']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['pre_exec']))

        if 'executable' in d:
            if isinstance(d['executable'], str) or isinstance(d['executable'], unicode):
                self._executable = d['executable']
            else:
                raise TypeError(expected_type=str,
                                actual_type=type(d['executable']))

        if 'arguments' in d:
            if isinstance(d['arguments'], list):
                self._arguments = d['arguments']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['arguments']))

        if 'post_exec' in d:
            if isinstance(d['post_exec'], list):
                self._post_exec = d['post_exec']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['post_exec']))

        if 'cpu_reqs' in d:
            if isinstance(d['cpu_reqs'], dict):
                self._cpu_reqs = d['cpu_reqs']
            else:
                raise TypeError(expected_type=dict,
                                actual_type=type(d['cpu_reqs']))

        if 'gpu_reqs' in d:
            if isinstance(d['gpu_reqs'], dict):
                self._gpu_reqs = d['gpu_reqs']
            else:
                raise TypeError(expected_type=dict,
                                actual_type=type(d['gpu_reqs']))

        if 'lfs_per_process' in d:
            if d['lfs_per_process']:
                if isinstance(d['lfs_per_process'], int):
                    self._lfs_per_process = d['lfs_per_process']
                else:
                    raise TypeError(expected_type=int,
                                    actual_type=type(d['lfs_per_process']))

        if 'upload_input_data' in d:
            if isinstance(d['upload_input_data'], list):
                self._upload_input_data = d['upload_input_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['upload_input_data']))

        if 'copy_input_data' in d:
            if isinstance(d['copy_input_data'], list):
                self._copy_input_data = d['copy_input_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['copy_input_data']))

        if 'link_input_data' in d:
            if isinstance(d['link_input_data'], list):
                self._link_input_data = d['link_input_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['link_input_data']))

        if 'move_input_data' in d:
            if isinstance(d['move_input_data'], list):
                self._move_input_data = d['move_input_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['move_input_data']))


        if 'copy_output_data' in d:
            if isinstance(d['copy_output_data'], list):
                self._copy_output_data = d['copy_output_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['copy_output_data']))

        if 'move_output_data' in d:
            if isinstance(d['move_output_data'], list):
                self._move_output_data = d['move_output_data']
            else:
                raise TypeError(expected_type=list,
                                actual_type=type(d['move_output_data']))

        if 'download_output_data' in d:
            if isinstance(d['download_output_data'], list):
                self._download_output_data = d['download_output_data']
            else:
                raise TypeError(expected_type=list, actual_type=type(
                    d['download_output_data']))

        if 'stdout' in d:
            if d['stdout']:
                if isinstance(d['stdout'], str) or isinstance(d['stdout'], unicode):
                    self._stdout = d['stdout']
                else:
                    raise TypeError(expected_type=str, actual_type=type(d['stdout']))

        if 'stderr' in d:
            if d['stderr']:
                if isinstance(d['stderr'], str) or isinstance(d['stderr'], unicode):
                    self._stderr = d['stderr']
                else:
                    raise TypeError(expected_type=str, actual_type=type(d['stderr']))

        if 'exit_code' in d:
            if d['exit_code']:
                if isinstance(d['exit_code'], int):
                    self._exit_code = d['exit_code']
                else:
                    raise TypeError(
                        entity='exit_code', expected_type=int, actual_type=type(d['exit_code']))

        if 'path' in d:
            if d['path']:
                if isinstance(d['path'], str) or isinstance(d['path'], unicode):
                    self._path = d['path']
                else:
                    raise TypeError(entity='path', expected_type=str,
                                    actual_type=type(d['path']))

        if 'tag' in d:
            if d['tag']:
                if isinstance(d['tag'], str) or isinstance(d['tag'], unicode):
                    self._tag = str(d['tag'])
                else:
                    raise TypeError(expected_type=str,
                                    actual_type=type(d['tag']))

        if 'parent_stage' in d:
            if isinstance(d['parent_stage'], dict):
                self._p_stage = d['parent_stage']
            else:
                raise TypeError(
                    entity='parent_stage', expected_type=dict, actual_type=type(d['parent_stage']))

        if 'parent_pipeline' in d:
            if isinstance(d['parent_pipeline'], dict):
                self._p_pipeline = d['parent_pipeline']
            else:
                raise TypeError(entity='parent_pipeline', expected_type=dict, actual_type=type(
                    d['parent_pipeline']))