"""Tests for academic data dictionary documentation and programmatic metadata."""

from pathlib import Path
import unittest
import pandas as pd

from src.data_dictionary import (
    ACADEMIC_DATA_DICTIONARY,
    get_data_dictionary,
    get_data_dictionary_dataframe,
    get_entity_fields,
    get_field_spec,
)


class TestDataDictionary(unittest.TestCase):
    """Test suite ensuring data dictionary completeness and analytical metadata integrity."""

    def setUp(self):
        """Set expected entities and mandatory attribute keys."""
        self.expected_entities = {
            "students",
            "courses",
            "enrollments",
            "attendance",
            "assignments",
            "submissions",
            "exams",
            "interventions",
        }
        self.mandatory_keys = {
            "meaning",
            "type",
            "source",
            "expected_values",
            "required",
            "analytical_usage",
        }

    def test_all_expected_entities_documented(self):
        """Verify all academic entities are defined in the programmatic dictionary."""
        dict_keys = set(ACADEMIC_DATA_DICTIONARY.keys())
        self.assertEqual(dict_keys, self.expected_entities)

    def test_field_attribute_completeness(self):
        """Verify every field has meaning, type, source, expected_values, required, and analytical_usage."""
        for entity_name, fields in ACADEMIC_DATA_DICTIONARY.items():
            self.assertGreater(len(fields), 0, f"Entity '{entity_name}' has no fields defined.")
            for field_name, attributes in fields.items():
                for key in self.mandatory_keys:
                    self.assertIn(
                        key,
                        attributes,
                        f"Missing '{key}' in entity '{entity_name}', field '{field_name}'",
                    )
                    self.assertIsNotNone(
                        attributes[key],
                        f"Empty '{key}' in entity '{entity_name}', field '{field_name}'",
                    )

    def test_specific_core_fields_present(self):
        """Verify core fields required by Concept #7 are present."""
        self.assertIsNotNone(get_field_spec("students", "student_id"))
        self.assertIsNotNone(get_field_spec("students", "name"))
        self.assertIsNotNone(get_field_spec("students", "program"))
        self.assertIsNotNone(get_field_spec("students", "year"))
        self.assertIsNotNone(get_field_spec("courses", "course_id"))
        self.assertIsNotNone(get_field_spec("attendance", "status"))
        self.assertIsNotNone(get_field_spec("assignments", "title"))
        self.assertIsNotNone(get_field_spec("submissions", "score"))
        self.assertIsNotNone(get_field_spec("exams", "score"))
        self.assertIsNotNone(get_field_spec("interventions", "notes"))

    def test_get_entity_fields(self):
        """Verify get_entity_fields returns a list of dictionaries with field_name."""
        fields = get_entity_fields("students")
        self.assertEqual(len(fields), 4)
        names = [f["field_name"] for f in fields]
        self.assertIn("student_id", names)
        self.assertIn("program", names)

    def test_get_data_dictionary_dataframe(self):
        """Verify DataFrame export of data dictionary for UI presentation."""
        df = get_data_dictionary_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 20)
        expected_cols = ["entity", "field_name", "type", "meaning", "source", "expected_values", "required", "analytical_usage"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Test single entity filter
        df_students = get_data_dictionary_dataframe("students")
        self.assertEqual(len(df_students), 4)
        self.assertTrue((df_students["entity"] == "students").all())

    def test_markdown_documentation_file_exists(self):
        """Verify that docs/DATA_DICTIONARY.md exists and contains markdown tables."""
        doc_path = Path(__file__).resolve().parent.parent / "docs" / "DATA_DICTIONARY.md"
        self.assertTrue(doc_path.exists())
        content = doc_path.read_text(encoding="utf-8")

        for entity in self.expected_entities:
            self.assertIn(f"Entity: `{entity}`", content, f"Missing documentation section for entity '{entity}'")


if __name__ == "__main__":
    unittest.main()
