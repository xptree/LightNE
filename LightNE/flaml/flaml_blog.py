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
from functools import partial

logger = logging.getLogger(__name__)
# to ignore sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logging.basicConfig(filename="./tune_blog.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "../data_bin"
ne_out = "blog.ne"
pro_out = "blog.pro"
label = f"{data_path}/blogcatalog.mat"
input_path = f"{data_path}/blogcatalog.adj"
order_range = 10

#'coeff0': 26.902187316024058, 'coeff1': 92.56102766639714, 'coeff2': 97.51983787610096, 'coeff3': 59.58175769452938, 'coeff4': 66.46687223346987, 'coeff5': 13.426494235512436, 'coeff6': 36.8395232112682, 'coeff7': 79.49616846973117, 'coeff8': 30.03171554095294, 'coeff9': 14.651426664258764, 'order': 11.0, 'theta': 0.45926465853589216, 'mu': 0.1672452776910997
#36.25 (0.27) & 38.89 (0.09) & 40.35 (0.23) & 41.50 (0.07) & 42.21 (0.15) & 42.76 (0.13) & 42.87 (0.42) & 43.55 (0.19) & 43.84 (0.68)
#22.70 (0.37) & 25.53 (0.10) & 27.22 (0.20) & 28.35 (0.31) & 29.16 (0.59) & 29.44 (0.51) & 29.95 (0.78) & 30.46 (0.15) & 29.80 (1.05)
config={
    	"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	"theta": tune.uniform(0.1, 1.0),   # consider quniform may be better
    	"mu": tune.uniform(0.1, 1.0),
    	"step0": tune.uniform(0, 1.0), #[tune.uniform(0.01, 1.0) for x in range(order_range)],
        "step1": tune.uniform(0, 1.0),
        "step2": tune.uniform(0, 1.0),
        "step3": tune.uniform(0, 1.0),
        "step4": tune.uniform(0, 1.0),
        "step5": tune.uniform(0, 1.0),
        "step6": tune.uniform(0, 1.0),
        "step7": tune.uniform(0, 1.0),
        "step8": tune.uniform(0, 1.0),
        "step9": tune.uniform(0, 1.0),
    }
#points_to_evaluate = [
#    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0},
#]
#evaluated_rewards = [26.85]

points_to_evaluate = [
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":0.0,"step3":0.0,"step4":0.0,"step5":0.0,"step6":0.0,"step7":0.0,"step8":0.0,"step9":0.0},
]
evaluated_rewards = [26.19]


search_alg = flaml.BlendSearch(space = config,
        metric = "macro_f1_mean",
        mode = "max",
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards
        #low_cost_partial_config = {"num_train_epochs":2}
        )

def objective(resource_attr,config):
    #step_coeff_list = config['step']
    resource = int(config[resource_attr])
    print("resource:",resource)
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
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 10000 -walklen 10 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 4096 -dim 128 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 1 -sample_ratio {resource} -upper 0 -tablesz 6679660000 -sparse_project 0 -power_iteration {q} -oversampling 50 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    macro_f1_mean = node_classification.main(['--label',label,'--embedding',pro_out,'--start-train-ratio','10','--stop-train-ratio','90','--num-train-ratio','9','--C','10','--num-split','3','--binary'])
    print("macro_f1_mean:",macro_f1_mean)
    #tune.report(macro_f1_mean=macro_f1_mean)
    return {resource_attr: resource, "macro_f1_mean": macro_f1_mean}


resource_attr = "sample_size"
np.random.seed(1)
analysis = tune.run(
    partial(objective, resource_attr),
    config = config,
    #evaluation_function=objective,
    #search_alg=flaml.CFO(
    #    space=config,
    #    metric="macro_f1_mean",
    #    mode="max",
    #    low_cost_partial_config={"num_train_epochs": 2}),
    #search_alg=search_alg,
    metric="macro_f1_mean",
    mode = "max",
    points_to_evaluate=points_to_evaluate,
    evaluated_rewards=evaluated_rewards,
    num_samples=100, 
    time_budget_s=360000, 
    resource_attr=resource_attr, 
    scheduler="flaml", 
    max_resource = 2000, 
    min_resource = 500, 
    reduction_factor = 100,
    metric_constraints=[("macro_f1_mean", ">=", 26.19)],
    use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


search_alg.save("./checkpoints/blog.tune")



