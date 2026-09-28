import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('analysis',ROOT/'scripts/analyze_recorded.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

class RecordedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs=json.loads((ROOT/'results/2026-09-28/runs.json').read_text(encoding='utf-8'))
    def test_archived_matrix(self):
        analysis.validate(self.runs)
        self.assertIn('149.974',analysis.render(self.runs))
        self.assertIn('3.211',analysis.render(self.runs))
    def test_rejects_missing_repeat(self):
        altered=copy.deepcopy(self.runs)
        altered['12700H-h3-cpu']['samples'].pop()
        with self.assertRaises(ValueError): analysis.validate(altered)
    def test_rejects_cuda_on_cpu(self):
        altered=copy.deepcopy(self.runs)
        altered['13900HX-h3-cpu']['cuda_initialized_final']=True
        with self.assertRaises(ValueError): analysis.validate(altered)
    def test_rejects_failed_run(self):
        altered=copy.deepcopy(self.runs)
        altered['14900HX-image-cuda']['status']='failed'
        with self.assertRaises(ValueError): analysis.validate(altered)

if __name__=='__main__': unittest.main()
