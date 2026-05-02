import unittest
from typing import Any
from unittest.mock import patch

from app.control.account.commands import AccountPatch
from app.control.account.enums import QuotaSource
from app.control.account.models import (
    AccountMutationResult,
    AccountRecord,
    QuotaWindow,
)
from app.control.account.quota_defaults import default_quota_set
from app.control.account.refresh import AccountRefreshService


class _FakeRepo:
    def __init__(self, records: list[AccountRecord]) -> None:
        self.records = {record.token: record for record in records}

    async def get_accounts(self, tokens: list[str]) -> list[AccountRecord]:
        return [self.records[token] for token in tokens if token in self.records]

    async def patch_accounts(
        self, patches: list[AccountPatch]
    ) -> AccountMutationResult:
        patched = 0
        for account_patch in patches:
            record = self.records.get(account_patch.token)
            if record is None:
                continue

            qs = record.quota_set()
            updates: dict[str, Any] = {}
            quota_changed = False

            for mode_id, field_name in (
                (0, "quota_auto"),
                (1, "quota_fast"),
                (2, "quota_expert"),
                (3, "quota_heavy"),
                (4, "quota_grok_4_3"),
            ):
                payload = getattr(account_patch, field_name)
                if payload is None:
                    continue
                qs.set(mode_id, QuotaWindow.from_dict(payload))
                quota_changed = True

            if quota_changed:
                updates["quota"] = qs.to_dict()
            if account_patch.pool is not None:
                updates["pool"] = account_patch.pool
            if account_patch.last_sync_at is not None:
                updates["last_sync_at"] = account_patch.last_sync_at
            if account_patch.usage_sync_delta is not None:
                updates["usage_sync_count"] = (
                    record.usage_sync_count + account_patch.usage_sync_delta
                )

            self.records[account_patch.token] = record.model_copy(update=updates)
            patched += 1

        return AccountMutationResult(patched=patched, revision=1)


def _make_record(token: str, pool: str) -> AccountRecord:
    return AccountRecord(
        token=token,
        pool=pool,
        quota=default_quota_set(pool).to_dict(),
    )


class AccountRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_on_import_promotes_basic_record_to_super(self) -> None:
        repo = _FakeRepo([_make_record("tok-basic", "basic")])
        service = AccountRefreshService(repo)
        fast_super = QuotaWindow(
            remaining=120,
            total=140,
            window_seconds=7200,
            reset_at=1234567890000,
            synced_at=1234567890000,
            source=QuotaSource.REAL,
        )

        async def fake_fetch_all_quotas(token: str, pool: str):
            self.assertEqual(token, "tok-basic")
            self.assertEqual(pool, "basic")
            return {1: fast_super}

        with patch.object(service, "_fetch_all_quotas", fake_fetch_all_quotas):
            result = await service.refresh_on_import(["tok-basic"])

        updated = repo.records["tok-basic"]
        quota = updated.quota_set()

        self.assertEqual(result.refreshed, 1)
        self.assertEqual(updated.pool, "super")
        self.assertEqual(quota.fast.total, 140)
        self.assertEqual(quota.fast.remaining, 120)
        self.assertEqual(quota.fast.source, QuotaSource.REAL)
        self.assertEqual(quota.auto.total, 50)
        self.assertEqual(quota.auto.remaining, 50)
        self.assertEqual(quota.auto.source, QuotaSource.DEFAULT)
        self.assertEqual(quota.expert.total, 50)
        self.assertEqual(quota.expert.remaining, 50)
        self.assertIsNotNone(quota.grok_4_3)
        self.assertEqual(quota.grok_4_3.total, 50)
        self.assertEqual(quota.grok_4_3.remaining, 50)

    async def test_refresh_tokens_keeps_super_pool_when_auto_window_is_missing(
        self,
    ) -> None:
        repo = _FakeRepo([_make_record("tok-super", "super")])
        service = AccountRefreshService(repo)
        fast_super = QuotaWindow(
            remaining=98,
            total=140,
            window_seconds=7200,
            reset_at=1234567890000,
            synced_at=1234567890000,
            source=QuotaSource.REAL,
        )

        async def fake_fetch_all_quotas(token: str, pool: str):
            self.assertEqual(token, "tok-super")
            self.assertEqual(pool, "super")
            return {1: fast_super}

        with patch.object(service, "_fetch_all_quotas", fake_fetch_all_quotas):
            result = await service.refresh_tokens(["tok-super"])

        updated = repo.records["tok-super"]
        quota = updated.quota_set()

        self.assertEqual(result.refreshed, 1)
        self.assertEqual(updated.pool, "super")
        self.assertEqual(quota.fast.total, 140)
        self.assertEqual(quota.fast.remaining, 98)
        self.assertEqual(quota.fast.source, QuotaSource.REAL)
        self.assertEqual(quota.auto.total, 50)
        self.assertEqual(quota.auto.remaining, 50)
        self.assertEqual(quota.auto.source, QuotaSource.DEFAULT)


if __name__ == "__main__":
    unittest.main()
