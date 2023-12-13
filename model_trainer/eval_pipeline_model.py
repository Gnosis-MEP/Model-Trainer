import json
import os
import glob
import sys
import statistics
import time

from PIL import Image

import cv2
import numpy as np

from model_trainer.conf import EVAL_CONFS_JSON


class PipelineEvaluator(object):

    def __init__(self):
        self.eval_confs = None
        self.setup_eval_confs()
        self.predicted_values = {}
        self.true_positives_events = []
        self.true_negatives_events = []
        self.false_positives_events = []
        self.false_negatives_events = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1score = 0
        self.latencies = []

    def setup_eval_confs(self):
        self.eval_confs = None
        with open(EVAL_CONFS_JSON, 'r') as f:
            self.eval_confs = json.load(f)

    def calculate_accuracy(self):
        true_positives = len(self.true_positives_events)
        true_negatives = len(self.true_negatives_events)
        false_positives = len(self.false_positives_events)
        false_negatives = len(self.false_negatives_events)

        nominator = true_positives + true_negatives
        denominator = true_positives + true_negatives + false_positives + false_negatives
        if denominator == 0:
            return 0
        accuracy = nominator / denominator
        return accuracy

    def calculate_precision(self):
        true_positives = len(self.true_positives_events)
        false_positives = len(self.false_positives_events)

        denominator = (true_positives + false_positives)
        if denominator == 0:
            return 0
        return true_positives / denominator

    def calculate_recall(self):
        true_positives = len(self.true_positives_events)
        false_negatives = len(self.false_negatives_events)
        denominator = (true_positives + false_negatives)
        if denominator == 0:
            return 0
        return true_positives / denominator

    def calculate_f1_score(self, precision, recall):
        nominator = 2 * (precision * recall)
        denominator = precision + recall
        if denominator == 0:
            return 0
        f_1 = nominator / denominator
        return f_1

    def calculate_latency(self):
        if len(self.latencies) == 0:
            avg_latency = 0
        else:
            avg_latency = statistics.mean(self.latencies)
        if len(self.latencies) > 1:
            std_latency = statistics.stdev(self.latencies)
        else:
            std_latency = 0
        return avg_latency, std_latency

    def calculate_metrics(self):
        self.accuracy = self.calculate_accuracy()
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f_score = self.calculate_f1_score(self.precision, self.recall)
        avg_latency, std_latency = self.calculate_latency()
        througput = 0
        if len(self.latencies) != 0:
            througput = 1 / avg_latency
        results = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f_score,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'throughput': througput,
        }
        return results

    def run(self, pipeline_prediction_json):
        with open(pipeline_prediction_json, 'r') as f:
            json_data = json.load(f)
            self.predicted_values = json_data['results']
            self.latencies = json_data['stats']['processing_times']

        total_to_eval = self.eval_confs['Total_Non_Ignored_Frames']
        total_images_predicted = len(self.predicted_values.keys())
        total_eval = 0
        sorted_images_ids = sorted(self.predicted_values.keys())
        for image_id in sorted_images_ids:

            has_oi = self.predicted_values[image_id]

            positive_pred = has_oi

            if image_id in self.eval_confs['TP']:
                if positive_pred:
                    self.true_positives_events.append(image_id)
                else:
                    self.false_negatives_events.append(image_id)
            elif image_id in self.eval_confs['TN']:
                if not positive_pred:
                    self.true_negatives_events.append(image_id)
                else:
                    self.false_positives_events.append(image_id)

            total_eval += 1

        if self.debug:
            print(f'total_eval: {total_eval}. total_to_eval {total_to_eval}. total_images_predicted: {total_images_predicted}.')
        results = self.calculate_metrics()
        return results

if __name__ == '__main__':
    pipeline_prediction_json = sys.argv[1]
    evaluator = PipelineEvaluator()
    evaluator.debug = True
    res = evaluator.run(pipeline_prediction_json)
    print(f'EVAL_CONFS_JSON: {EVAL_CONFS_JSON}')
    print(json.dumps(res, indent=4))
