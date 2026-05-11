"""機械学習手法のIDの定数定義モジュール"""

# csvファイル保存用手法名
LOGISTIC_REGRESSION = "logreg"
RANDOM_FOREST = "rf"
GRADIENT_BOOSTING_DECISION_TREE = "gbdt"
XGBOOST = "xgboost"

# データセットのパス
PATH_TRAIN = "data\\titanic\\train.csv"
PATH_TEST = "data\\titanic\\test.csv"


# Config file output path
CONFIG_FILE_EXTENSION = ".yaml"
CONFIG_FOLDER_PREFIX = ".\\output\\config\\case"

CONFIG_FILE_PREFIX_XGBOOST = "config_case"

SAVE_MODEL_FILE_PREFIX_XGBOOST = "case_"
SAVE_MODEL_FILE_EXTENSION_XGBOOST = ".json"

SAVE_MODEL_ROOT_XGBOOST = ".\\model\\"
SAVE_MODEL_FILE_PARENT_XGBOOST = "\\case_"
