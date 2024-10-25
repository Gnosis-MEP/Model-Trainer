import os

from decouple import config

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)

REDIS_ADDRESS = config('REDIS_ADDRESS', default='localhost')
REDIS_PORT = config('REDIS_PORT', default='6379')

TRACER_REPORTING_HOST = config('TRACER_REPORTING_HOST', default='localhost')
TRACER_REPORTING_PORT = config('TRACER_REPORTING_PORT', default='6831')

SERVICE_STREAM_KEY = config('SERVICE_STREAM_KEY')


# DATASET_ID = config('DATASET_ID', default='TS-D-Q-1-5S')
# DATASET_ID = config('DATASET_ID', default='TS-D-Q-1-10S')
# DATASET_ID = config('DATASET_ID', default='TS-D-Q-2-10S')
# DATASET_ID = config('DATASET_ID', default='TS-D-Q-1b-10S')
# DATASET_ID = config('DATASET_ID', default='TS-D-B-2-10S')
DATASET_ID = config('DATASET_ID', default='HS-D-B-1-10S')

AUGMENTED_DATASETS_PATH = config('AUGMENTED_DATASETS_PATH')

DATASET_PATH = config('DATASET_PATH', default=os.path.join(AUGMENTED_DATASETS_PATH, DATASET_ID))
MODELS_PATH = config('MODELS_PATH', default=os.path.join(PROJECT_ROOT, 'data', 'models'))
# MODEL_ID =  config('MODEL_ID', default=f'{DATASET_ID}_-150_car_person-bird-dog')
# MODEL_ID =  config('MODEL_ID', default=f'{DATASET_ID}_-300_person_car-bird-dog')
# MODEL_ID =  config('MODEL_ID', default=f'{DATASET_ID}_-300_car_person-bird-dog_region')
# MODEL_ID =  config('MODEL_ID', default=f'{DATASET_ID}_-300_car-person_bird-dog')
MODEL_ID =  config('MODEL_ID', default=f'{DATASET_ID}_-300_car')


# MODEL_ID =  config('MODEL_ID', default=f'TS-D-Q-1b-10S_-300_car_person-bird-dog') # overriding just to test out a model in another dataset


EVAL_PATH = config('EVAL_PATH', default=os.path.join(PROJECT_ROOT, 'data', 'eval'))
EVAL_ID = MODEL_ID

# EVAL_ID = 'TS-D-Q-1_TS-D-Q-1b-10S_-300_car_person-bird-dog' # overriding just to test out a model in another dataset

EVAL_CONFS_JSON = config('EVAL_CONFS_JSON', default=os.path.join(EVAL_PATH, f'{EVAL_ID}.json'))

EVAL_PREDICTION_JSON = config('EVAL_PREDICTION_JSON', default=os.path.join(EVAL_PATH, f'pred-{EVAL_ID}.json'))
EVAL_DIFF_PREDICTION_JSON = config('EVAL_DIFF_PREDICTION_JSON', default=os.path.join(EVAL_PATH, f'pred_diff_{{t}}_-{EVAL_ID}.json'))

# LISTEN_EVENT_TYPE_SOME_EVENT_TYPE = config('LISTEN_EVENT_TYPE_SOME_EVENT_TYPE')
# LISTEN_EVENT_TYPE_OTHER_EVENT_TYPE = config('LISTEN_EVENT_TYPE_OTHER_EVENT_TYPE')

SERVICE_CMD_KEY_LIST = [
    # LISTEN_EVENT_TYPE_SOME_EVENT_TYPE,
    # LISTEN_EVENT_TYPE_OTHER_EVENT_TYPE,
]

# PUB_EVENT_TYPE_NEW_EVENT_TYPE = config('PUB_EVENT_TYPE_NEW_EVENT_TYPE')

PUB_EVENT_LIST = [
    # PUB_EVENT_TYPE_NEW_EVENT_TYPE,
]

# Only for Content Extraction services
SERVICE_DETAILS = None

# Example of how to define SERVICE_DETAILS from env vars:
# SERVICE_DETAILS_SERVICE_TYPE = config('SERVICE_DETAILS_SERVICE_TYPE')
# SERVICE_DETAILS_STREAM_KEY = config('SERVICE_DETAILS_STREAM_KEY')
# SERVICE_DETAILS_QUEUE_LIMIT = config('SERVICE_DETAILS_QUEUE_LIMIT', cast=int)
# SERVICE_DETAILS_THROUGHPUT = config('SERVICE_DETAILS_THROUGHPUT', cast=float)
# SERVICE_DETAILS_ACCURACY = config('SERVICE_DETAILS_ACCURACY', cast=float)
# SERVICE_DETAILS_ENERGY_CONSUMPTION = config('SERVICE_DETAILS_ENERGY_CONSUMPTION', cast=float)
# SERVICE_DETAILS_CONTENT_TYPES = config('SERVICE_DETAILS_CONTENT_TYPES', cast=Csv())
# SERVICE_DETAILS = {
#     'service_type': SERVICE_DETAILS_SERVICE_TYPE,
#     'stream_key': SERVICE_DETAILS_STREAM_KEY,
#     'queue_limit': SERVICE_DETAILS_QUEUE_LIMIT,
#     'throughput': SERVICE_DETAILS_THROUGHPUT,
#     'accuracy': SERVICE_DETAILS_ACCURACY,
#     'energy_consumption': SERVICE_DETAILS_ENERGY_CONSUMPTION,
#     'content_types': SERVICE_DETAILS_CONTENT_TYPES
# }

LOGGING_LEVEL = config('LOGGING_LEVEL', default='DEBUG')