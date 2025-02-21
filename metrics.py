from rouge import Rouge
import numpy as np
import re

class LegalMetrics:
    def __init__(self):
        self.rouge = Rouge()
        self.citation_pattern = r'\[CITATION\]'
        self.section_pattern = r'\[SECTION\]'

    def compute(self, preds, targets):
        results = {}
        
        # ROUGE Scores
        rouge_scores = self.rouge.get_scores(preds, targets, avg=True)
        results['rouge1'] = rouge_scores['rouge-1']['f']
        results['rouge2'] = rouge_scores['rouge-2']['f']
        results['rougeL'] = rouge_scores['rouge-l']['f']

        # Legal-specific metrics
        results['citation_recall'] = self._citation_recall(preds, targets)
        results['section_coverage'] = self._section_coverage(preds, targets)
        
        return results

    def _citation_recall(self, preds, targets):
        recalls = []
        for pred, target in zip(preds, targets):
            target_cits = set(re.findall(self.citation_pattern, target))
            pred_cits = set(re.findall(self.citation_pattern, pred))
            if len(target_cits) > 0:
                recalls.append(len(pred_cits & target_cits) / len(target_cits))
        return np.mean(recalls) if recalls else 0.0

    def _section_coverage(self, preds, targets):
        coverages = []
        for pred, target in zip(preds, targets):
            target_sections = set(re.findall(self.section_pattern, target))
            pred_sections = set(re.findall(self.section_pattern, pred))
            if len(target_sections) > 0:
                coverages.append(len(pred_sections & target_sections) / len(target_sections))
        return np.mean(coverages) if coverages else 0.0