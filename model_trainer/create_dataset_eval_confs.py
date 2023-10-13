import json
import os
import sys


from model_trainer.dataloader import get_loader
from model_trainer.conf import EVAL_PATH, DATASET_ID


# def get_total_ois_and_others_cls_for_annotation(annotation, ois_cls_ids, other_classes_ids):
#     total_ois = 0
#     total_other = 0
#     for oi_id in ois_cls_ids:
#         for obj in annotation['data']:
#             if obj['class_id'] == oi_id:
#                 total_ois += 1
#             elif obj['class_id'] in other_classes_ids:
#                 total_other += 1

#     return total_ois, total_other

def get_total_ois_and_others_cls_for_annotation(annotation, ois_cls_ids, other_classes_ids):
    ois_set = set()
    others_set = set()
    for obj in annotation['data']:
        if obj['class_id'] in ois_cls_ids:
            ois_set.add(obj['class_id'])
        elif obj['class_id'] in other_classes_ids:
            others_set.add(obj['class_id'])

    total_ois = len(ois_set)
    total_other = len(others_set)

    return total_ois, total_other


def create_segment_eval_config(base_annotations_json_path, query_ois, other_classes, n_ignored_frames):
    base_annotations = None
    with open(base_annotations_json_path, 'r') as f:
        base_annotations = json.load(f)

    eval_conf_name = f'{DATASET_ID}_-{n_ignored_frames}_{"-".join(query_ois)}_{"-".join(other_classes)}'

    ois_cls_ids = [
        int(k) for k, v in base_annotations['classes'].items() if v['name'].lower() in query_ois
    ]
    other_classes_ids =  [
        int(k) for k, v in base_annotations['classes'].items() if v['name'].lower() in other_classes
    ]
    print(ois_cls_ids)

    partial_ois = []
    other_objects = []

    true_positives = []
    true_negatives = []
    total_non_ignored_frames = len(base_annotations['annotations'].keys()) - n_ignored_frames
    for example_id, annotation in base_annotations['annotations'].items():
        frame_number = int(example_id.split('frame_')[1])
        if frame_number <= n_ignored_frames:
            continue
        total_ois, total_others = get_total_ois_and_others_cls_for_annotation(annotation, ois_cls_ids, other_classes_ids)
        if total_ois == len(ois_cls_ids):
            true_positives.append(example_id)
        else:
            true_negatives.append(example_id)
            if total_ois > 0:
                partial_ois.append(example_id)
            if total_others > 0:
                other_objects.append(example_id)

    segment_eval_config = {
        'Total_Non_Ignored_Frames': total_non_ignored_frames,
        'SizePerc': {
            'TP': (len(true_positives), len(true_positives)/total_non_ignored_frames),
            'TN': (len(true_negatives), len(true_negatives)/total_non_ignored_frames),
            'Partials': (len(partial_ois), len(partial_ois)/total_non_ignored_frames),
            'others': (len(other_objects), len(other_objects)/total_non_ignored_frames),
        },
        'Classes_Examined':  query_ois + other_classes,
        'OIs_id': ois_cls_ids,
        'Others_id': other_classes_ids,
        'TP': true_positives,
        'TN': true_negatives,
        'Partial_OIs': partial_ois,
        'Other_Objects': other_objects,
    }

    evan_conf_path = os.path.join(EVAL_PATH, f'{eval_conf_name}.json')
    with open(evan_conf_path, 'w') as f:
        json.dump(segment_eval_config, f)


    print(json.dumps(segment_eval_config, indent=4))
    print(f'total: {total_non_ignored_frames}')
    print(f'TP: {len(true_positives)} ({len(true_positives)/total_non_ignored_frames * 100}%)')
    print(f'TN: {len(true_negatives)} ({len(true_negatives)/total_non_ignored_frames * 100}%)')
    print(f'Partials: {len(partial_ois)} ({len(partial_ois)/total_non_ignored_frames * 100}%)')
    print(f'others: {len(other_objects)} ({len(other_objects)/total_non_ignored_frames * 100}%)')

if __name__ == '__main__':
    # python model_trainer/create_dataset_eval_confs.py ../live-street-datasets/my-creations/selected/Annotations/TS-D-Q-1/base/annotations_TS-D-Q-1.json 5 person,car dog
    base_annotations_json_path = sys.argv[1]
    n_ignored_sec = int(sys.argv[2])
    query_ois = sys.argv[3].split(',')
    other_classes = sys.argv[4].split(',')

    n_ignored_frames = 30 * n_ignored_sec



    create_segment_eval_config(base_annotations_json_path, query_ois, other_classes, n_ignored_frames)
