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
import h5py

logger = logging.getLogger(__name__)
# to ignore sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logging.basicConfig(filename="./tune_hl2014.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')


data_path = "/home/jiezhong/dev/graph_embedding/data_bin"
ne_out = "hl2014.ne"
pro_out = ""
input_path = f"{data_path}/hyperlink2014_sym.64.bytepda.train"
order_range = 2

#'coeff0': 49.40656309959399, 'coeff1': 94.60900198650822
#113.21736370129355 0.4270637560038328 0.3729069121498294 0.5268232679946036 0.6613364018728672 0.8869534187132986

config={
    	#"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	#"theta": tune.uniform(0.1, 1.0),   # consider quniform may be better
    	#"mu": tune.uniform(0.1, 1.0),
    	"step0": tune.uniform(1, 100.0), #[tune.uniform(0.01, 1.0) for x in range(order_range)],
        "step1": tune.uniform(1, 100.0),
        "q": tune.randint(1,4),
    }
points_to_evaluate = [
    {"step0":1.0,"step1":1.0,"q":3},
]
evaluated_rewards = [1.116182]

#search_alg = flaml.BlendSearch(space = config,
#        metric = "metrics_mean",
#        mode = "max",
#        points_to_evaluate=points_to_evaluate,
#        evaluated_rewards=evaluated_rewards
        #low_cost_partial_config = {"num_train_epochs":2}
#        )

def objective(config):
    #step_coeff_list = config['step']
    step_coeff=""
    for i in range(order_range):
        s = "step"+str(i)
        if i == order_range - 1:
            step_coeff = step_coeff + str(config[s])
        else:
            step_coeff = step_coeff + str(config[s]) + ","
    order = 10 #config['order']
    theta = 0.5 #config['theta']
    mu = 0.2 #config['mu']
    q = config['q']
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 1 -walklen 2 -step_coeff {step_coeff} -rounds 1 -s -m -c -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 32 -dim 32 -order 10 -theta 0.5 -mu 0.2 -analyze 1 -sample 1 -sample_ratio 0.05 -mem_ratio 0.125 -upper 0 -tablesz 1549475257 -sparse_project 0 -power_iteration {q} -oversampling 10 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    metrics_mean = lp_ranking.main(['--links',f"{data_path}/hyperlink2014_sym.edges.test",'--embedding',ne_out,'--binary','--norm','--dim','32','--num-uniform-neg','1000','--num-degree-neg','0'])
    print(metrics_mean)
    tune.report(metrics_mean=metrics_mean)

np.random.seed(1)
analysis = tune.run(
    evaluation_function=objective,
    #search_alg=flaml.CFO(
    #    space=config,
    #    metric="macro_f1_mean",
    #    mode="max",
    #    low_cost_partial_config={"num_train_epochs": 2}),
    # search_alg=search_alg,
    config = config,
    metric = "metrics_mean",
    mode="max",
    points_to_evaluate=points_to_evaluate,
    evaluated_rewards=evaluated_rewards,
    num_samples=20, use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


#search_alg.save("./checkpoints/hl2014.tune")



