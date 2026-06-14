import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric import ed25519

from core.enterprise_policy import EnterprisePolicy
from core.license import (
    authorization_from_hardware_and_license,
    canonical_payload,
    hardware_fingerprint,
    verify_license_file,
)


class LicenseTests(unittest.TestCase):
    def test_verify_signed_license_bound_to_hardware(self):
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes_raw()
        payload = {
            "license_id": "lic-1",
            "subject": "ACME Lab",
            "level": "ENTERPRISE",
            "hardware_hash": hardware_fingerprint("USB-SN-1"),
            "expires_at": int(time.time()) + 3600,
            "features": ["rag", "audit"],
        }
        signature = private_key.sign(canonical_payload(payload))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "license.json"
            path.write_text(json.dumps({
                "payload": payload,
                "signature": base64.b64encode(signature).decode("ascii"),
            }), encoding="utf-8")

            status = verify_license_file(
                path,
                hardware_sn="USB-SN-1",
                require_signature=True,
                public_key_b64=base64.b64encode(public_key_bytes).decode("ascii"),
            )

        self.assertTrue(status.valid)
        self.assertEqual(status.level, "ENTERPRISE")
        self.assertEqual(status.subject, "ACME Lab")

    def test_license_rejects_hardware_mismatch(self):
        payload = {
            "level": "PRO",
            "hardware_hash": hardware_fingerprint("OTHER-SN"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "license.json"
            path.write_text(json.dumps({"payload": payload}), encoding="utf-8")

            status = verify_license_file(path, hardware_sn="USB-SN-1", require_signature=False)

        self.assertFalse(status.valid)
        self.assertEqual(status.reason, "license hardware mismatch")

    def test_policy_can_disable_lite_fallback_when_license_missing(self):
        policy = EnterprisePolicy(allow_lite_mode=False)
        with patch("core.license.find_license_path", return_value=None):
            auth_level, _, status = authorization_from_hardware_and_license(
                hardware_level="LITE",
                hardware_sn="USB-SN-1",
                workspace_path=None,
                policy=policy,
            )

        self.assertEqual(auth_level, "NONE")
        self.assertEqual(status.reason, "lite mode disabled by enterprise policy")


if __name__ == "__main__":
    unittest.main()
