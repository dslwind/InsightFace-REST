import os
import unittest
from unittest.mock import patch

from if_rest.settings import Models, Settings


class SettingsTests(unittest.TestCase):
    def test_ga_model_env_alias_is_supported(self):
        with patch.dict(os.environ, {"GA_MODEL": "genderage_v1"}, clear=False):
            models = Models()
            self.assertEqual(models.ga_name, "genderage_v1")

    def test_settings_build_nested_models_from_current_env(self):
        with patch.dict(os.environ, {"GA_NAME": "genderage_v1"}, clear=False):
            settings = Settings()
            self.assertEqual(settings.models.ga_name, "genderage_v1")

    def test_models_dir_env_override(self):
        with patch.dict(os.environ, {"MODELS_DIR": "/tmp/custom-models"}, clear=False):
            settings = Settings()
            self.assertEqual(settings.models_dir, "/tmp/custom-models")

    def test_root_images_dir_env_override(self):
        with patch.dict(os.environ, {"ROOT_IMAGES_DIR": "/tmp/custom-images"}, clear=False):
            settings = Settings()
            self.assertEqual(settings.root_images_dir, "/tmp/custom-images")


if __name__ == "__main__":
    unittest.main()
