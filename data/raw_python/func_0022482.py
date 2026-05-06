def luid(self):
        """
        Unique ID of the current stage (fully qualified).

        example:
            >>> stage.luid
            pipe.0001.stage.0004

        :getter: Returns the fully qualified uid of the current stage
        :type: String
        """
        p_elem = self.parent_pipeline.get('name')
        if not p_elem:
            p_elem = self.parent_pipeline['uid']

        s_elem = self.name
        if not s_elem:
            s_elem = self.uid

        return '%s.%s' % (p_elem, s_elem)