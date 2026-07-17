import os
import sys
import tempfile
import unittest
from unittest.mock import patch

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.config import config


class TestSaveConfigBackup(unittest.TestCase):
    """save_config must snapshot the previous on-disk config to .bak BEFORE
    overwriting — config.toml is gitignored, so that backup is the only
    recovery path when a bad in-memory state gets persisted."""

    def test_previous_config_is_backed_up_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_config = os.path.join(tmp, "config.toml")
            with open(fake_config, "w", encoding="utf-8") as f:
                f.write('[app]\npexels_api_keys = "PRECIOUS-KEY"\n')

            with patch.object(config, "config_file", fake_config):
                config.save_config()

            with open(fake_config + ".bak", encoding="utf-8") as f:
                backup = f.read()
            self.assertIn("PRECIOUS-KEY", backup)

    def test_missing_config_file_saves_without_backup_or_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_config = os.path.join(tmp, "config.toml")
            with patch.object(config, "config_file", fake_config):
                config.save_config()
            self.assertTrue(os.path.isfile(fake_config))
            self.assertFalse(os.path.isfile(fake_config + ".bak"))

    def test_identical_resave_does_not_rotate_the_backup(self):
        """The WebUI saves on every rerun; unchanged content must be a no-op
        so a good .bak survives until the config actually changes again."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_config = os.path.join(tmp, "config.toml")
            with open(fake_config, "w", encoding="utf-8") as f:
                f.write('[app]\npexels_api_keys = "PRECIOUS-KEY"\n')

            with patch.object(config, "config_file", fake_config):
                config.save_config()  # rotates: .bak = PRECIOUS-KEY version
                first_bak_mtime = os.path.getmtime(fake_config + ".bak")
                config.save_config()  # identical content -> must not touch disk
                config.save_config()
            self.assertEqual(
                os.path.getmtime(fake_config + ".bak"), first_bak_mtime
            )
            with open(fake_config + ".bak", encoding="utf-8") as f:
                self.assertIn("PRECIOUS-KEY", f.read())

    def test_backup_failure_never_blocks_the_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_config = os.path.join(tmp, "config.toml")
            with open(fake_config, "w", encoding="utf-8") as f:
                f.write("[app]\n")
            with patch.object(config, "config_file", fake_config), patch.object(
                config.shutil, "copyfile", side_effect=OSError("disk full")
            ):
                config.save_config()
            # the save itself still happened
            with open(fake_config, encoding="utf-8") as f:
                self.assertIn("[app]", f.read())


if __name__ == "__main__":
    unittest.main()
