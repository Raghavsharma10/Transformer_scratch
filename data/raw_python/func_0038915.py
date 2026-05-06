def attend_pendings(self):
        """
        Check all chats created with no agent assigned yet.
        Schedule a timer timeout to call it.
        """
        chats_attended = []
        pending_chats = Chat.pending.all()
        for pending_chat in pending_chats:
            free_agent = self.strategy.free_agent() 
            if free_agent:
                pending_chat.attend_pending(free_agent, self)
                pending_chat.save()
                chats_attended.append(pending_chat)
            else:
                break
        return chats_attended