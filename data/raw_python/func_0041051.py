def compile(self, source_code, post_treatment=''.join, source='<string>', target='exec'): 
        """Return ready-to-exec object code of compilation.
        Use python built-in compile function.
        Use exec(1) on returned object for execute it."""
        self.last_python_code = super().compile(source_code, post_treatment)
        return PyCompiler.executable(self.last_python_code, source, target)