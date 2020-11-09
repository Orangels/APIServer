import traceback

import numpy as np

Min_L2 = 8
Max_L2 = 16
OUT_SHAPE = 515
NUM_CLASSES = 512


def get_age_gender_info(age_gender):
    features = []
    norms = []
    ages = []
    genders = []
    rescores = []

    try:

        if [age[0] for age in age_gender]:
            for info in age_gender:

                # 年龄
                age = info[512]
                ages = np.append(ages, age)

                # 性别
                gender = info[513:515]
                gender_prob = np.exp(gender[0]) / (np.exp(gender[0]) + np.exp(gender[1]))
                genders = np.append(genders, gender_prob)

                # rescore
                if OUT_SHAPE == 517:
                    rescore = info[515:517]
                    rescore_prob = np.exp(rescore[1]) / (np.exp(rescore[0]) + np.exp(rescore[1]))
                    rescores = np.append(rescores, rescore_prob)
                else:
                    rescore = np.array([0, 0])
                    rescore_prob = np.exp(rescore[1]) / (np.exp(rescore[0]) + np.exp(rescore[1]))
                    rescores = np.append(rescores, rescore_prob)

                # norm, feature
                # TODO 使用真实np格式
                outputs = age_gender[:, 0:512]
                norm, feature = l2_norm(outputs)
                features = np.append(features, feature)
                norms = np.append(norms, norm)
            features = features.reshape([-1, NUM_CLASSES])
            norms = np.asarray(norms).reshape([-1, 1])
            norms = (norms - Min_L2) / (Max_L2 - Min_L2)
        return ages, genders, norms, features, rescores
    except Exception as e:
        print(traceback.format_exc())


def l2_norm(input, axis=1):
    norm = np.linalg.norm(input, ord=2, axis=axis)
    output = ((input.T) / norm).T
    return norm, output
