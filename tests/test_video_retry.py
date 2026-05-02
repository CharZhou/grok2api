import asyncio
import types
import unittest
from unittest.mock import patch

from app.platform.errors import UpstreamError
from app.products.openai import video


class _FakeDirectory:
    def __init__(self) -> None:
        self.released: list[str] = []
        self.feedback_calls: list[tuple[str, object, int]] = []

    async def release(self, lease) -> None:
        self.released.append(lease.token)

    async def feedback(self, token: str, kind, mode_id: int) -> None:
        self.feedback_calls.append((token, kind, mode_id))


class _FakeConfig:
    def get_float(self, key: str, default: float = 0.0) -> float:
        return 180.0 if key == "video.timeout" else default


class _FakeSpec:
    mode_id = 0

    def is_video(self) -> bool:
        return True


class VideoRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = _FakeDirectory()
        self.spec = _FakeSpec()
        self.fail_sync_calls: list[tuple[str, int, str | None]] = []
        self.quota_sync_calls: list[tuple[str, int]] = []

    def _patch_common(self):
        async def _fail_sync(token: str, mode_id: int, exc=None) -> None:
            status = getattr(exc, "status", None)
            self.fail_sync_calls.append((token, mode_id, None if status is None else str(status)))

        async def _quota_sync(token: str, mode_id: int) -> None:
            self.quota_sync_calls.append((token, mode_id))

        return patch.multiple(
            video,
            get_config=lambda: _FakeConfig(),
            resolve_model=lambda model: self.spec,
            selection_max_retries=lambda: 1,
            _configured_retry_codes=lambda cfg: frozenset({429}),
            _should_retry_upstream=lambda exc, retry_codes: getattr(exc, "status", None) in retry_codes,
            _fail_sync=_fail_sync,
            _quota_sync=_quota_sync,
            now_s=lambda: 123,
        )

    async def test_video_retry_switches_account_and_retries(self) -> None:
        reserve_calls: list[list[str]] = []
        attempt_calls: list[tuple[int, int]] = []
        leases = [types.SimpleNamespace(token="tok-1"), types.SimpleNamespace(token="tok-2")]

        async def fake_reserve_account(directory, spec, *, now_s_override=None, exclude_tokens=None):
            reserve_calls.append(list(exclude_tokens or []))
            idx = len(reserve_calls) - 1
            return leases[idx], 0

        async def runner(token: str, timeout_s: float) -> str:
            if token == "tok-1":
                raise UpstreamError("Video upstream returned 429", status=429, body="rate limited")
            return f"ok:{token}:{int(timeout_s)}"

        async def attempt_started(attempt: int, total_attempts: int) -> None:
            attempt_calls.append((attempt, total_attempts))

        with self._patch_common(), patch("app.dataplane.account._directory", self.directory), patch.object(video, "reserve_account", fake_reserve_account):
            result = await video._run_video_with_account(
                model="grok-imagine-video",
                runner=runner,
                attempt_started_cb=attempt_started,
            )
            await asyncio.sleep(0)

        self.assertEqual(result, "ok:tok-2:180")
        self.assertEqual(reserve_calls, [[], ["tok-1"]])
        self.assertEqual(attempt_calls, [(1, 2), (2, 2)])
        self.assertEqual(self.directory.released, ["tok-1", "tok-2"])
        self.assertEqual(self.fail_sync_calls, [("tok-1", 0, "429")])
        self.assertEqual(self.quota_sync_calls, [("tok-2", 0)])

    async def test_video_retry_exhausted_adds_attempt_context(self) -> None:
        reserve_calls: list[list[str]] = []
        leases = [types.SimpleNamespace(token="tok-1"), types.SimpleNamespace(token="tok-2")]

        async def fake_reserve_account(directory, spec, *, now_s_override=None, exclude_tokens=None):
            reserve_calls.append(list(exclude_tokens or []))
            idx = len(reserve_calls) - 1
            return leases[idx], 0

        async def runner(token: str, timeout_s: float) -> str:
            raise UpstreamError("Video upstream returned 429", status=429, body="rate limited")

        with self._patch_common(), patch("app.dataplane.account._directory", self.directory), patch.object(video, "reserve_account", fake_reserve_account):
            with self.assertRaises(UpstreamError) as ctx:
                await video._run_video_with_account(
                    model="grok-imagine-video",
                    runner=runner,
                )
            await asyncio.sleep(0)

        self.assertEqual(reserve_calls, [[], ["tok-1"]])
        self.assertIn("attempt 2/2", ctx.exception.message)
        self.assertIn("retried 1 time(s)", ctx.exception.message)
        self.assertEqual(self.directory.released, ["tok-1", "tok-2"])

    async def test_video_non_retryable_error_fails_immediately(self) -> None:
        reserve_calls: list[list[str]] = []
        leases = [types.SimpleNamespace(token="tok-1")]

        async def fake_reserve_account(directory, spec, *, now_s_override=None, exclude_tokens=None):
            reserve_calls.append(list(exclude_tokens or []))
            return leases[0], 0

        async def runner(token: str, timeout_s: float) -> str:
            raise UpstreamError("Video upstream returned 500", status=500, body="server error")

        with self._patch_common(), patch("app.dataplane.account._directory", self.directory), patch.object(video, "reserve_account", fake_reserve_account):
            with self.assertRaises(UpstreamError) as ctx:
                await video._run_video_with_account(
                    model="grok-imagine-video",
                    runner=runner,
                )
            await asyncio.sleep(0)

        self.assertEqual(reserve_calls, [[]])
        self.assertEqual(ctx.exception.status, 500)
        self.assertNotIn("attempt 2/2", ctx.exception.message)
        self.assertEqual(self.directory.released, ["tok-1"])


if __name__ == "__main__":
    unittest.main()
