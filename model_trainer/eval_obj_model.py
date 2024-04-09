import json
import os
import glob
import sys

from matplotlib import pyplot

import torch
import torchvision.models as models

from fastai.basics import default_device
from fastai.vision.all import load_learner, Path, PILImage, vision_learner, accuracy, show_image

from model_trainer.dataloader import get_loader
from model_trainer.conf import DATASET_PATH, EVAL_CONFS_JSON, EVAL_PREDICTION_JSON, DATASET_ID


class ClsEvaluator(object):

    def __init__(self, eval_images_dir, base_model, fine_tune_name):
        self.eval_confs = None
        self.setup_eval_confs()
        self.eval_images_dir = eval_images_dir
        self.images_paths = glob.glob(os.path.join(eval_images_dir, '*.png'))
        try:
            dls = get_loader(DATASET_PATH)
            self.learn = vision_learner(dls, base_model, metrics=accuracy)
            self.learn.load(fine_tune_name)
            self.learn.model.cuda()
        except FileNotFoundError:
            # if already run predictions, and is now using the pre-saved prediction data, and already deleted the
            # dataset from local disk
            pass
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
        accuracy = nominator / denominator
        return accuracy

    def calculate_precision(self):
        true_positives = len(self.true_positives_events)
        false_positives = len(self.false_positives_events)

        return true_positives / (true_positives + false_positives)

    def calculate_recall(self):
        true_positives = len(self.true_positives_events)
        false_negatives = len(self.false_negatives_events)
        return true_positives / (true_positives + false_negatives)

    def calculate_f1_score(self, precision, recall):
        nominator = 2 * (precision * recall)
        denominator = precision + recall
        f_1 = nominator / denominator
        return f_1

    def calculate_metrics(self):
        self.accuracy = self.calculate_accuracy()
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f_score = self.calculate_f1_score(self.precision, self.recall)
        results = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f_score,
        }
        return results

    def run(self, threshold=0.5, recreate_predict=False):
        run_predict = not os.path.exists(EVAL_PREDICTION_JSON) or recreate_predict
        if run_predict is False:
            with open(EVAL_PREDICTION_JSON, 'r') as f:
                self.predicted_values = json.load(f)

        total_to_eval = self.eval_confs['Total_Non_Ignored_Frames']
        total_images = len(self.images_paths)
        total_eval = 0
        for proc_index, image_path in enumerate(self.images_paths):
            image_id = os.path.basename(image_path).split('.')[0]

            if run_predict is True:
                class_probs = self.model_predict(image_path)
                self.predicted_values[image_id] = class_probs
            else:
                class_probs = self.predicted_values[image_id]

            non_oi_prob, has_oi_prob = class_probs

            positive_pred = has_oi_prob > threshold

            processed = 1
            if image_id in self.eval_confs['TP']:
                if positive_pred:
                    self.true_positives_events.append(image_id)
                else:
                    self.false_positives_events.append(image_id)
            elif image_id in self.eval_confs['TN']:
                if not positive_pred:
                    self.true_negatives_events.append(image_id)
                else:
                    self.false_negatives_events.append(image_id)
            else:
                processed = 0

            total_eval += processed
            if self.debug and proc_index % 500 == 0:
                print(f'i: {proc_index + 1}/{total_images}. total_eval: {total_eval/total_to_eval * 100}%')

        if run_predict is True:
            # with open(EVAL_PREDICTION_JSON, 'w') as f:
            #     json.dump(self.predicted_values, f)
            pass

        if self.debug:
            print(f'total_eval: {total_eval}. total_to_eval {total_to_eval}. total_images: {total_images}.')
        results = {'threshold': threshold}
        results.update(self.calculate_metrics())
        return results

    def model_predict(self, image_path):
        # img = PILImage.create(image_path)

        # Make predictions on the image
        predicted_class, _, class_probs = self.learn.predict(image_path)

        # Print the predicted class
        # print(f'Predicted class: {predicted_class}')

        return class_probs.tolist()


# def model_predict(model, image_path):

#     dls = get_loader(DATASET_PATH)
#     learn = vision_learner(dls, models.mobilenet_v3_large, metrics=accuracy)
#     learn.load('cls_model')
#     learn.model.cuda()

#     img = PILImage.create(image_path)

#     # Make predictions on the image
#     predicted_class, _, class_probs = learn.predict(image_path)

#     # Print the predicted class
#     print(f'Predicted class: {predicted_class}')

#     return predicted_class.tolist()
#     # # Print the confidence scores for each class
#     # for class_name, prob in zip(learn.dls.vocab, class_probs):
#     #     print(f'Class: {class_name}, Confidence Score: {prob}')

#     # show_image(img)
#     # pyplot.show()


if __name__ == '__main__':
    # threshold = float(sys.argv[1])
    raise Exception('use the eval from the pytorch obj, this was discontinued and never changed from the CLS model')

    print(f'EVAL_CONFS_JSON: {EVAL_CONFS_JSON}')
    print(f'EVAL_PREDICTION_JSON: {EVAL_PREDICTION_JSON}')
    eval_images_dir = f'/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/{DATASET_ID.split("-10S")[0]}'
    evaluator = ClsEvaluator(eval_images_dir, models.mobilenet_v3_large, 'cls_model')
    evaluator.debug = False
    ths = [0.5, 0.8, 0.95]
    # ths = [0.5,]
    all_res = {}
    for threshold in ths:
        res = evaluator.run(threshold=threshold, recreate_predict=False)
        res['frames_dropped'] = len(evaluator.true_negatives_events) + len(evaluator.false_negatives_events)
        res['frames_dropped_perc'] = res['frames_dropped'] / evaluator.eval_confs['Total_Non_Ignored_Frames']
        all_res[str(threshold)] = res
    print(json.dumps(all_res, indent=4))
    # evaluator.save()


