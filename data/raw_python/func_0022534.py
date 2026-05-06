def luid(self):
        """
        Unique ID of the current task (fully qualified).

        example:
            >>> task.luid
            pipe.0001.stage.0004.task.0234

        :getter: Returns the fully qualified uid of the current task
        :type: String
        """
        p_elem = self.parent_pipeline.get('name')
        if not p_elem:
            p_elem = self.parent_pipeline['uid']

        s_elem = self.parent_stage.get('name')
        if not s_elem:
            s_elem = self.parent_stage['uid']

        t_elem = self.name
        if not t_elem:
            t_elem = self.uid

        return '%s.%s.%s' % (p_elem, s_elem, t_elem)