import unittest

from app.control.model.enums import ModeId, Tier
from app.control.model.registry import get


class ModelRegistryTests(unittest.TestCase):
    def test_video_model_uses_auto_super_selection_defaults(self) -> None:
        spec = get("grok-imagine-video")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.mode_id, ModeId.AUTO)
        self.assertEqual(spec.tier, Tier.SUPER)
        self.assertEqual(spec.pool_candidates(), (1, 2))


if __name__ == "__main__":
    unittest.main()
