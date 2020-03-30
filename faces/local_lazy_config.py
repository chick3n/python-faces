"""
Lazy Config Override
"""

import os as _os
import confuse as _confuse

class LocalLazyConfig(_confuse.LazyConfig):
    """
    LocalLazyConfig
    """
    def config_dir(self):
        """
        config_dir
        """
        local_config = _os.path.join(_os.getcwd(), _confuse.CONFIG_FILENAME)
        if _os.path.exists(local_config):
            return _os.getcwd()

        return super(LocalLazyConfig, self).__init__(self.appname)
