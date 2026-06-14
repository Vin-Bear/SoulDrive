import unittest

from core.parent_child_index import split_parent_document


class ParentChildIndexTests(unittest.TestCase):
    def test_split_parent_document_preserves_parent_id_and_offsets(self):
        text = "A" * 120 + "B" * 120 + "C" * 120

        children = split_parent_document(
            parent_id="paper-parent-1",
            text=text,
            child_size=140,
            child_overlap=20,
        )

        self.assertEqual(children[0].parent_id, "paper-parent-1")
        self.assertEqual(children[0].start_char, 0)
        self.assertGreater(len(children), 1)
        self.assertLess(children[1].start_char, children[0].end_char)


if __name__ == "__main__":
    unittest.main()
