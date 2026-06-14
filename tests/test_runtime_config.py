import os
import unittest
from unittest.mock import patch

from core.runtime_config import api_port, api_base_url


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_uses_env_port(self):
        with patch.dict(os.environ, {"SOULDRIVE_API_PORT": "19091"}):
            self.assertEqual(api_port(), 19091)
            self.assertEqual(api_base_url(), "http://127.0.0.1:19091")


if __name__ == "__main__":
    unittest.main()
