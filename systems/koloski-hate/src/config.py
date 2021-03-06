PATH_DATA = "../../../datasets/pan21-hate-train"
PATH_DATA_EN = "../../../datasets/pan21-hate-train/en"
PATH_DATA_ES = "../../../datasets/pan21-hate-train/es"
PATH_DATA_EN_TRUTH = "../../../datasets/pan21-hate-train/en/truth.txt"
PATH_DATA_ES_TRUTH = "../../../datasets/pan21-hate-train/es/truth.txt"
PATH_EXPR = "../expr"
PATH_MODELS = "../models"
PATH_OUT = "../../../output/koloski-hate"
PATH_IMGS = "../imgs"
# D2V
D2V_EPOCHS = 50
D2V_LR = 0.01
D2V_VS = 100
# TPOT
TPOT_OUT = 2

# EXTENDED_TFIDF
parametersWide = {"loss": ["hinge", "log", "modified_huber"], \
                  "penalty": ["elasticnet"], "alpha": [0.01, 0.001, 0.0001, 0.0005], \
                  "l1_ratio": [0.05, 0.25, 0.3, 0.6, 0.8, 0.95], \
                  "power_t": [0.5, 0.1, 0.9]}
# TIRA_ONE_PARAMTERES
parametersTIRA_DIMS = [256, 512, 768]
parametersTIRA_FEATURES = [2500, 5000, 10000, 15000]
parametersTIRA_GS1 = {"loss": ["hinge", "log"], \
                      "penalty": ["elasticnet"], \
                      "alpha": [0.01, 0.001, 0.0001, 0.0005], \
                      "l1_ratio": [0.05, 0.25, 0.3, 0.6, 0.8, 0.95], \
                      "power_t": [0.5, 0.1, 0.9]}
parametersTIRA_GS2 = {"C": [0.1, 1, 10, 25, 50, 100, 500], "penalty": ["l2"]}
