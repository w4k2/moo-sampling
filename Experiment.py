from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score

from algorithm import MooHcsSelection
from imblearn.over_sampling import RandomOverSampler
from smote_variants import SMOTE, Borderline_SMOTE2, ADASYN, NoSMOTE
import pandas as pd
import numpy as np
import algorithm
import os
import argparse

from sklearn.base import clone
from DatasetsCollection import load

import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger("Maciek")

metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
criteria = ['best_precision', 'best_recall', 'balanced']
algorithms = ['best_precision', 'best_recall', 'balanced']
reference_models = [RandomOverSampler, SMOTE, Borderline_SMOTE2, ADASYN, NoSMOTE]
file_list = [
             "page-blocks-1-3_vs_4", "yeast-0-5-6-7-9_vs_4", "yeast-1-2-8-9_vs_7",
             "yeast-1-4-5-8_vs_7", "yeast-1_vs_7", "yeast-2_vs_4", "yeast-2_vs_8", "yeast4", "yeast5", "yeast6",
             "ecoli-0-1-4-7_vs_2-3-5-6",
              "ecoli-0-1_vs_2-3-5", "ecoli-0-2-6-7_vs_3-5",
             "ecoli-0-6-7_vs_3-5", "ecoli-0-6-7_vs_5",
             "yeast-0-2-5-6_vs_3-7-8-9", "yeast-0-3-5-9_vs_7-8",
             "abalone-17_vs_7-8-9-10", "abalone-19_vs_10-11-12-13",
             "abalone-20_vs_8-9-10", "flare-F", "kr-vs-k-zero_vs_eight",
             "poker-8-9_vs_5", "poker-8-9_vs_6", "poker-8_vs_6",
             "winequality-red-4",
              "winequality-white-3-9_vs_5", "winequality-white-3_vs_7",
             "ecoli1", "ecoli2", "ecoli3", "glass0", "glass1", "haberman",
             "pima", "yeast3"
]


def experiment_parallel(data,fold_no, classifier, path="experiment"):
    try:
        os.mkdir(path)
    except:
        pass

    fold = data[fold_no]
    i = fold_no

    fold_path = os.path.join(path, "fold" + str(i))
    try:
        os.mkdir(fold_path)
    except:
        pass

    logger.info("Starting sampling the data")

    moo_sampler = algorithm.MooHcsSelection(classifier, measures=[precision_score, recall_score], criteria=['best', 'balanced'])
    resampled_data = moo_sampler.fit_sample(fold[0][0], fold[0][1])

    for model in reference_models:
        resampled_data.append(model.fit_sample(fold[0][0], fold[0][1]))

    logger.info("Finished sampling the data")

    models = criteria + [str(m) for m in reference_models]

    for ia in range(len(models)):
        data = resampled_data[ia]
        if data is not None:
            c = clone(classifier)
            c.fit(data[0], data[1])
            y_pred = c.predict(fold[1][0])
            df = pd.DataFrame(y_pred)
            df.to_csv(os.path.join(fold_path, "classifier_prediction_" + models[ia] + ".csv"))
        else:
            logger.warning("Resampled data is empty")
    return


def conduct_experiment(path="experiments-no-radii-no-bound-full3", dataset='all', fold=10):
    try:
        logger.info(f"Starting experiment on dataset {dataset} on {fold} fold")

        data_set = []
        if dataset == 'all':
            names = file_list
            for file in file_list:
                data_set.append(load(file, transformed=transformed))

        else:
            names = [dataset]
            data_set.append(load(dataset))

        try:
            os.mkdir(path)
        except:
            pass

        classifier = DecisionTreeClassifier(random_state=7, criterion="gini")

        try:
            os.mkdir(os.path.join(path, "results"))
        except:
            pass

        for i, data in enumerate(data_set):
            if fold < 10:
                experiment_parallel(data, fold, classifier, os.path.join(path,"results", names[i]))
            else:
                for i in range(10):
                    experiment_parallel(data, classifier, os.path.join(path,"results", names[i]))

    except Exception as e:
        logger.warning(f"{dataset} FOLD NO {fold} EXCEPTION {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--fold", type=int, default=10)
    args = parser.parse_args()
    conduct_experiment(args.result_directory, args.dataset, args.fold)
