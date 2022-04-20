import numpy as np
import scipy
import subprocess
import logging
import numpy as np
import time
import flaml
from flaml import tune
import os
import node_classification
import lp_ranking
import lp

logger = logging.getLogger(__name__)
# to ignore sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logging.basicConfig(filename="./tune_pld.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "/home/jiezhong/dev/graph_embedding/data_bin"
ne_out = "hyperlink.ne"
pro_out = "hyperlink.pro"
input_path = f"{data_path}/hyperlink.adj"
order_range = 5

#'coeff0': 38.73642149524113, 'coeff1': 0.2251243250997348, 'coeff2': 94.36805501530901, 'coeff3':73.57994906198027, 'coeff4': 99.05427540864785, 'order': 11.0, 'theta': 0.9935742923133584, 'mu': 0.10994955421734764
#0.969456705525866

config={
    	"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	"theta": tune.uniform(0.1, 1.0),   # consider quniform may be better
    	"mu": tune.uniform(0.1, 1.0),
    	"step0": tune.uniform(0.01, 1.0), #[tune.uniform(0.01, 1.0) for x in range(order_range)],
        "step1": tune.uniform(0.01, 1.0),
        "step2": tune.uniform(0.01, 1.0),
        "step3": tune.uniform(0.01, 1.0),
        "step4": tune.uniform(0.01, 1.0),
    }
points_to_evaluate = [
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0,"step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0},
]
evaluated_rewards = [96.7]

search_alg = flaml.BlendSearch(space = config,
        metric = "auc",
        mode = "max",
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards
        #low_cost_partial_config = {"num_train_epochs":2}
        )

def objective(config):
    #step_coeff_list = config['step']
    step_coeff=""
    for i in range(order_range):
        s = "step"+str(i)
        if i == order_range - 1:
            step_coeff = step_coeff + str(config[s])
        else:
            step_coeff = step_coeff + str(config[s]) + ","
    order = config['order']
    theta = config['theta']
    mu = config['mu']
    q = 1
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 10 -walklen 5 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 128 -dim 128 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 1 -sample_ratio 15 -normalize 1 -upper 0 -tablesz 15494752578 -sparse_project 0 -power_iteration {q} -oversampling 10 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    auc = lp.main(['--idmap','./hyperlink.idmap.npy','--embedding',pro_out,'--binary'])
    print(auc)
    tune.report(auc=auc)

np.random.seed(10)
analysis = tune.run(
    evaluation_function=objective,
    #search_alg=flaml.CFO(
    #    space=config,
    #    metric="macro_f1_mean",
    #    mode="max",
    #    low_cost_partial_config={"num_train_epochs": 2}),
    search_alg=search_alg,
    num_samples=30, use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


search_alg.save("./checkpoints/pld.tune")



