import importlib
import unittest
from unittest.mock import patch

from starlette.datastructures import FormData

import app.products.openai.video as video
from app.platform.errors import ValidationError

openai_router = importlib.import_module("app.products.openai.router")


class VideoPromptArrayTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_video_prompts_accepts_prompt_array_field(self) -> None:
        form = FormData(
            [
                ("prompt[]", "segment one"),
                ("prompt[]", "segment two"),
                ("prompt[]", "segment three"),
            ]
        )

        prompts = openai_router._extract_video_prompts(form)

        self.assertEqual(
            prompts,
            ["segment one", "segment two", "segment three"],
        )

    def test_extract_video_prompts_accepts_json_array_string(self) -> None:
        form = FormData([("prompt", '["segment one","segment two"]')])

        prompts = openai_router._extract_video_prompts(form)

        self.assertEqual(prompts, ["segment one", "segment two"])

    def test_resolve_segment_prompts_requires_matching_segment_count(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            video._resolve_segment_prompts(
                ["segment one", "segment two"],
                seconds=30,
            )

        self.assertEqual(ctx.exception.param, "prompt")
        self.assertIn("segment count (3)", ctx.exception.message)

    async def test_generate_video_with_token_uses_prompt_per_segment(self) -> None:
        create_post_prompts: list[str] = []
        segment_payloads: list[dict] = []

        async def fake_create_media_post(
            token: str,
            media_type: str,
            media_url: str = "",
            prompt: str = "",
            referer: str = "https://grok.com/imagine",
        ) -> dict:
            create_post_prompts.append(prompt)
            return {"post": {"id": "parent-post-id"}}

        async def fake_collect_video_segment(
            *,
            token: str,
            payload: dict,
            referer: str,
            timeout_s: float,
            progress_cb=None,
        ):
            segment_payloads.append(payload)
            index = len(segment_payloads)
            return video._VideoArtifact(
                video_url=f"https://assets.grok.com/video-{index}.mp4/content",
                video_post_id=f"video-post-{index}",
                asset_id=f"asset-{index}",
                thumbnail_url="",
            )

        with (
            patch.object(video, "create_media_post", fake_create_media_post),
            patch.object(video, "_collect_video_segment", fake_collect_video_segment),
        ):
            artifact = await video._generate_video_with_token(
                token="tok",
                prompt=["segment one", "segment two", "segment three"],
                aspect_ratio="16:9",
                resolution_name="720p",
                seconds=30,
                preset="normal",
                timeout_s=180.0,
            )

        self.assertEqual(create_post_prompts, ["segment one"])
        self.assertEqual(len(segment_payloads), 3)
        self.assertEqual(segment_payloads[0]["message"], "segment one --mode=normal")
        self.assertEqual(segment_payloads[1]["message"], "segment two --mode=normal")
        self.assertEqual(segment_payloads[2]["message"], "segment three --mode=normal")
        self.assertEqual(
            segment_payloads[1]["responseMetadata"]["modelConfigOverride"]["modelMap"][
                "videoGenModelConfig"
            ]["originalPrompt"],
            "segment two",
        )
        self.assertEqual(
            segment_payloads[2]["responseMetadata"]["modelConfigOverride"]["modelMap"][
                "videoGenModelConfig"
            ]["originalPrompt"],
            "segment three",
        )
        self.assertEqual(artifact.video_post_id, "video-post-3")


if __name__ == "__main__":
    unittest.main()
