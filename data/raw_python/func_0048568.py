def get_learning_path_session(self):
        """Gets the OsidSession associated with the learning path service.

        return: (osid.learning.LearningPathSession) - a
                LearningPathSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_learning_path() is false
        compliance: optional - This method must be implemented if
                    supports_learning_path() is true.

        """
        if not self.supports_learning_path():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.LearningPathSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session