def main(cls):
        """Setuptools console-script entrypoint"""
        cmd = cls()
        cmd._parse_args()
        cmd._setup_logging()
        response = cmd._run()
        output = cmd._handle_response(response)
        if output is not None:
            print(output)