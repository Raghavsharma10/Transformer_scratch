def create(self, message, mid=None, age=60, force=True):
        """
        create session
            force if you pass `force = False`, it may raise SessionError
                due to duplicate message id
        """
        with self.session_lock:
            if not hasattr(message, "id"):
                message.__setattr__("id", "event-%s" % (uuid.uuid4().hex,))
            if self.session_list.get(message.id, None) is not None:
                if force is False:
                    raise SessionError("Message id: %s duplicate!" %
                                       message.id)
                else:
                    message = Message(message.to_dict(), generate_id=True)

            session = {
                "status": Status.CREATED,
                "message": message,
                "age": age,
                "mid": mid,
                "created_at": time(),
                "is_published": Event(),
                "is_resolved": Event()
            }
            self.session_list.update({
                message.id: session
            })

            return session