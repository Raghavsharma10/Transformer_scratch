def get_payload(self):
        """Return Payload."""
        # Session id
        ret = bytes([self.session_id >> 8 & 255, self.session_id & 255])
        ret += bytes([self.originator.value])
        ret += bytes([self.priority.value])
        ret += bytes([0])  # ParameterActive pointing to main parameter (MP)
        # FPI 1+2
        ret += bytes([0])
        ret += bytes([0])

        # Main parameter + functional parameter
        ret += bytes(self.parameter)
        ret += bytes(32)

        # Nodes array: Number of nodes + node array + padding
        ret += bytes([len(self.node_ids)])  # index array count
        ret += bytes(self.node_ids) + bytes(20-len(self.node_ids))

        # Priority  Level Lock
        ret += bytes([0])
        # Priority Level information 1+2
        ret += bytes([0, 0])
        # Locktime
        ret += bytes([0])
        return ret