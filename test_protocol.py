import unittest

from experiment import load_config, purged_walk_forward_folds


class ProtocolTests(unittest.TestCase):
    def test_safe_folds_are_ordered_and_obey_embargo(self):
        config = load_config()
        embargo = int(config["embargo_rows"])
        folds = purged_walk_forward_folds(1000, config)
        self.assertEqual(len(folds), int(config["walk_forward_folds"]))
        for fold in folds:
            self.assertLess(fold.train.max(), fold.test.min())
        for previous, later in zip(folds, folds[1:]):
            embargoed = set(range(previous.test.max() + 1, previous.test.max() + 1 + embargo))
            self.assertTrue(embargoed.isdisjoint(set(later.train)))
            self.assertTrue(embargoed.isdisjoint(set(later.test)))


if __name__ == "__main__":
    unittest.main()
