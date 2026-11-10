import time
import unittest

from hermes_aipass.accounts import Account, AccountPool, AllAccountsUnavailableError


class TestAccountPool(unittest.TestCase):
    def test_single_account_behavior(self):
        pool = AccountPool.from_tokens(["tok_single"])
        self.assertEqual(len(pool.accounts), 1)
        acct = pool.get_account()
        self.assertEqual(acct.token, "tok_single")
        self.assertTrue(acct.is_available())

    def test_round_robin_selection(self):
        pool = AccountPool.from_tokens(["tok_1", "tok_2", "tok_3"])
        a1 = pool.get_account()
        a2 = pool.get_account()
        a3 = pool.get_account()
        a4 = pool.get_account()
        self.assertEqual(a1.token, "tok_1")
        self.assertEqual(a2.token, "tok_2")
        self.assertEqual(a3.token, "tok_3")
        self.assertEqual(a4.token, "tok_1")

    def test_429_exponential_cooldown_and_failover(self):
        pool = AccountPool.from_tokens(["tok_1", "tok_2"])
        a1 = pool.get_account()
        self.assertEqual(a1.token, "tok_1")

        # Report 429 on a1
        a1.report_429(base_cooldown_sec=10.0)
        self.assertFalse(a1.is_available())
        self.assertEqual(a1.consecutive_429s, 1)
        self.assertTrue(a1.cooldown_until > time.time())

        # Next call should failover to a2
        a2 = pool.get_account()
        self.assertEqual(a2.token, "tok_2")

    def test_all_accounts_dead_or_cooling_down_raises(self):
        pool = AccountPool.from_tokens(["tok_1", "tok_2"])
        for acct in pool.accounts:
            acct.report_dead("revoked")

        with self.assertRaises(AllAccountsUnavailableError):
            pool.get_account(wait_timeout=0.0)

    def test_report_ok_resets_streak(self):
        acct = Account(id=1, token="tok_test")
        acct.report_429(base_cooldown_sec=5.0)
        self.assertEqual(acct.consecutive_429s, 1)
        acct.report_ok()
        self.assertEqual(acct.consecutive_429s, 0)
        self.assertTrue(acct.is_available())

