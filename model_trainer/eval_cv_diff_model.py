import json
import os
import glob
import statistics
import time

from PIL import Image

import cv2
import numpy as np


from model_trainer.cv_diff_model_transforms import get_transforms
from model_trainer.model_cv_diff import get_base_diff_model
from model_trainer.conf import EVAL_CONFS_JSON, EVAL_DIFF_PREDICTION_JSON, MODEL_ID


class DiffEvaluator(object):

    def __init__(self, eval_images_dir, base_model, fine_tune_name):
        self.eval_confs = None
        self.setup_eval_confs()
        self.eval_images_dir = eval_images_dir
        self.images_paths = glob.glob(os.path.join(eval_images_dir, '*.png'))
        self.setup_model(base_model, fine_tune_name)
        self.predicted_values = {}
        self.true_positives_events = []
        self.true_negatives_events = []
        self.false_positives_events = []
        self.false_negatives_events = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1score = 0
        self.debug = False
        self.latencies = []

    def setup_model(self, base_model, fine_tune_name):
        self.model = base_model()
        self._hot_start(240, 240)

    def _hot_start(self, width, height):
        print('hotstart!')
        # Create black blank image
        image = Image.new("RGB", (width, height))
        self.model_predict(image, ignore=True)
        self.model.prev_key_frame = None


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

    def run(self, threshold=0.5, recreate_predict=False):
        self.model.diff_threshold = threshold

        t_predict_json = EVAL_DIFF_PREDICTION_JSON.format(t=int(threshold*100))

        run_predict = not os.path.exists(t_predict_json) or recreate_predict
        if run_predict is False:
            with open(t_predict_json, 'r') as f:
                self.predicted_values = json.load(f)

        total_to_eval = self.eval_confs['Total_Non_Ignored_Frames']
        total_images = len(self.images_paths)
        total_eval = 0
        sorted_images_path = sorted(self.images_paths, key=lambda k: int(os.path.basename(k).split('.png')[0].split('frame_')[1]))
        for proc_index, image_path in enumerate(sorted_images_path):
            image_id = os.path.basename(image_path).split('.')[0]

            if run_predict is True:
                diff_perc = self.model_predict(image_path)

                if recreate_predict is True:
                    self.predicted_values[image_id] = diff_perc
            else:
                diff_perc = self.predicted_values[image_id]


            positive_pred = diff_perc > threshold

            processed = 1
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
            else:
                processed = 0

            total_eval += processed
            if self.debug and proc_index % 500 == 0:
                print(f'i: {proc_index + 1}/{total_images}. total_eval: {total_eval/total_to_eval * 100}%')
                print(f'{self.calculate_metrics()}')

        if recreate_predict is True:
            with open(t_predict_json, 'w') as f:
                json.dump(self.predicted_values, f)
        if self.debug:
            print(f'total_eval: {total_eval}. total_to_eval {total_to_eval}. total_images: {total_images}.')
        results = {'threshold': threshold}
        results.update(self.calculate_metrics())
        return results


    def model_predict(self, image_path, ignore=False):
        if isinstance(image_path, str):
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            image = image_path
        # input_tensor = torch.from_numpy(np.asarray(image))
        # preprocessing = get_transforms('VAL')
        # input_batch = preprocessing(image).unsqueeze(0)
        input_batch = image
        start_time = time.time()
        prediciton = self.model.predict(input_batch)
        perc_diff = prediciton
        end_time = time.time()
        predict_time = (end_time - start_time)
        if ignore is False:
            self.latencies.append(predict_time)


        return perc_diff

if __name__ == '__main__':
    # threshold = float(sys.argv[1])
    eval_images_dir = '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-Q-1'

    # # Disable logging for fastai and its dependencies
    # logging.getLogger('fastai').setLevel(logging.CRITICAL)
    # logging.getLogger('torch').setLevel(logging.CRITICAL)
    # logging.getLogger('fastai').setLevel(logging.CRITICAL)
    # logging.getLogger('torchvision').setLevel(logging.CRITICAL)


    base_model = get_base_diff_model()
    # threshold = 0.5
    # t_predict_json = EVAL_DIFF_PREDICTION_JSON.format(t=int(threshold*100))
    # if not os.path.exists(t_predict_json):
    #     evaluator = DiffEvaluator(eval_images_dir, base_model, MODEL_ID)
    #     evaluator.debug = True
    #     res = evaluator.run(threshold=threshold, recreate_predict=True)
    #     print(json.dumps(res, indent=4))
    res_list = []
    for threshold in [0.05, 0.10, 0.15, 0.25, 0.35, 0.5]:
        t_predict_json = EVAL_DIFF_PREDICTION_JSON.format(t=int(threshold*100))
        if not os.path.exists(t_predict_json):
            evaluator = DiffEvaluator(eval_images_dir, base_model, MODEL_ID)
            evaluator.debug = True
            res = evaluator.run(threshold=threshold, recreate_predict=True)
        res_list.append(res)
        time.sleep(5)
    print(json.dumps(res_list, indent=4))
