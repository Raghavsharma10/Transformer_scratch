def after_flush_postexec(self, session, context):
        """
        Event listener to recursively expire `left` and `right` attributes the
        parents of all modified instances part of this flush.
        """
        instances = self.instances[session]
        while instances:
            instance = instances.pop()
            if instance not in session:
                continue
            parent = self.get_parent_value(instance)

            while parent != NO_VALUE and parent is not None:
                instances.discard(parent)
                session.expire(parent, ['left', 'right', 'tree_id', 'level'])
                parent = self.get_parent_value(parent)
            else:
                session.expire(instance, ['left', 'right', 'tree_id', 'level'])
                self.expire_session_for_children(session, instance)