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

logger = logging.getLogger(__name__)
# to ignore sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logging.basicConfig(filename="./tune_friendster.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "/home/jiezhong/dev/graph_embedding/data_bin"
ne_out = "friendster.ne"
pro_out = "friendster.pro"
label = f"{data_path}/friendster.label.npy"
input_path = f"{data_path}/friendster.adj"
order_range = 1

#'coeff0': 73.75590544809988, 'order': 10.0, 'theta': 0.9053273494295828, 'mu': 0.10594088907336975
#79.06 (0.00) & 87.17 (0.00) & 90.75 (0.00) & 91.12 (0.00) & 91.31 (0.00) & 91.49 (0.00) & 91.87 (0.00) & 92.01 (0.00) & 92.16 (0.00) & 92.31 (0.00)
#73.62 (0.00) & 85.60 (0.00) & 91.18 (0.00) & 91.54 (0.00) & 91.77 (0.00) & 92.00 (0.00) & 92.43 (0.00) & 92.52 (0.00) & 92.66 (0.00) & 92.77 (0.00)


config={
    	"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	"theta": tune.uniform(0.1, 1.0),   # consider quniform may be better
    	"mu": tune.uniform(0.1, 1.0),
    	"step0": tune.uniform(0.01, 1.0), #[tune.uniform(0.01, 1.0) for x in range(order_range)],
    }
points_to_evaluate = [
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0},
]
evaluated_rewards = [89.35]

search_alg = flaml.BlendSearch(space = config,
        metric = "macro_f1_mean",
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
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 1 -walklen 1 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 96 -dim 96 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 1 -sample_ratio 20 -upper 0 -normalize 1 -tablesz 14444268540 -sparse_project 0 -power_iteration {q} -oversampling 10 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    macro_f1_mean = node_classification.main(['--label',label,'--embedding',pro_out,'--start-train-ratio','1','--stop-train-ratio','10','--num-train-ratio','10','--C','50','--num-split','3','--dim','96','--binary','--partial'])
    print(macro_f1_mean)
    tune.report(macro_f1_mean=macro_f1_mean)

np.random.seed(10)
analysis = tune.run(
    evaluation_function=objective,
    #search_alg=flaml.CFO(
    #    space=config,
    #    metric="macro_f1_mean",
    #    mode="max",
    #    low_cost_partial_config={"num_train_epochs": 2}),
    search_alg=search_alg,
    num_samples=20, use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


search_alg.save("./checkpoints/friendster.tune")



