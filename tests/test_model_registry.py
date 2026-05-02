import unittest

from app.control.model.enums import ModeId, Tier
from app.control.model.registry import get


class ModelRegistryTests(unittest.TestCase):
    def test_video_model_uses_all_pool_selection_defaults(self) -> None:
        spec = get("grok-imagine-video")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.mode_id, ModeId.FAST)
        self.assertEqual(spec.tier, Tier.BASIC)
        self.assertEqual(spec.pool_candidates(), (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
