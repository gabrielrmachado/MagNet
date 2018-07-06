## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST
from utils import prepare_data
from worker import AEDetector, DBDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator
import utils
import numpy as np


detector_I = AEDetector("./defensive_models/MNIST_I", p=1)
detector_II = AEDetector("./defensive_models/MNIST_II", p=1)
reformer = SimpleReformer("./defensive_models/MNIST_I")

id_reformer = IdReformer()
classifier = Classifier("./models/example_classifier")

detector_dict = dict()
detector_dict["I"] = detector_I
detector_dict["II"] = detector_II

dataset = MNIST()
operator = Operator(dataset, classifier, detector_dict, reformer)

idx = utils.load_obj("example_idx")
_, _, Y = prepare_data(MNIST(), idx[:2000])

# f = "mnist_test_set_deepfool.pkl"
f = "example_carlini_10.0.pkl"
# Y = np.argmax(dataset.test_labels[:2000], axis=1)

testAttack = AttackData(f, Y)
testAttack.data = testAttack.data[:2000]

evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate={"I": 0.01, "II": 0.01})

