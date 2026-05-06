def main():
    """MAIN"""
    config = {
        "api": {
            "services": [
                {
                    "name": "my_api",
                    "testkey": "testval",
                },
            ],
            "calls": {
                "hello_world": {
                    "delay": 5,
                    "priority": 1,
                    "arguments": None,
                },
                "marco": {
                    "delay": 1,
                    "priority": 1,
                },
                "pollo": {
                    "delay": 1,
                    "priority": 1,
                },
            }
        }
    }
    app = AppBuilder([MyAPI], Strategy(Print()), AppConf(config))
    app.run()