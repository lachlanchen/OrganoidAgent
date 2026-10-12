import unittest

import app


class MorphologyRequestTests(unittest.TestCase):
    def test_neo_workflow_accepts_jpeg_and_uses_dedicated_runner(self):
        workflow = app.MORPHOLOGY_WORKFLOW_ADAPTERS["neo_cpsam"]
        self.assertEqual(workflow["dataset_dirs"], {"07_NEO"})
        self.assertIn(".jpg", workflow["extensions"])
        self.assertEqual(workflow["entrypoint"], "scripts/run_neo_cpsam_morphology.py")

    def test_dataset_paths_accept_relative_and_absolute_inputs(self):
        relative = "03_Y-27632_experiment_10x/example.tif"
        absolute = app.DATA_DIR / relative
        self.assertEqual(app.safe_dataset_path(relative), absolute.resolve())
        self.assertEqual(app.safe_dataset_path(absolute), absolute.resolve())

    def test_dataset_paths_reject_absolute_inputs_outside_datasets(self):
        with self.assertRaisesRegex(ValueError, "inside datasets"):
            app.safe_dataset_path("/etc/passwd")

    def test_overlay_report_request_does_not_enable_extra_outputs(self):
        options = app.morphology_output_options(
            "Generate only segmentation overlay images and the analysis report."
        )
        self.assertTrue(options["overlays"])
        self.assertTrue(options["report"])
        self.assertFalse(options["statistics_figures"])
        self.assertFalse(options["comparison_gallery"])
        self.assertFalse(options["object_crops"])
        self.assertFalse(options["diagnostic_images"])
        self.assertFalse(options["results_csv"])

    def test_chinese_negative_statistics_request_is_respected(self):
        options = app.morphology_output_options(
            "只输出分割overlay和分析报告，不用生成其他图和线性统计图。"
        )
        self.assertFalse(options["statistics_figures"])

    def test_statistics_figure_requires_an_explicit_positive_request(self):
        options = app.morphology_output_options("生成叠加图、报告和统计图。")
        self.assertTrue(options["statistics_figures"])


if __name__ == "__main__":
    unittest.main()
