class ConfigSingleton:
    _instance = None

    def __new__(cls, config=None):
        if not cls._instance:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            cls._instance.config = config
        elif config:
            cls._instance.config = config  # Update the configuration
        return cls._instance

    @classmethod
    def get_config(cls):
        if not cls._instance or not cls._instance.config:
            raise ValueError("Configuration has not been initialized.")
        return cls._instance.config
