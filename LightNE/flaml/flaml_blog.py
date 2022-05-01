import numpy as np
import scipy
import subprocess
import logging
import numpy as np
import time
import flaml
from flaml import tune
from flaml import BlendSearch
import os
import node_classification

logger = logging.getLogger(__name__)
# to ignore sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logging.basicConfig(filename="./flaml_blog.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "../../data_bin"
ne_out = "blog.ne"
pro_out = "blog.pro"
label = f"{data_path}/blogcatalog.mat"
input_path = f"{data_path}/blogcatalog.adj"
order_range = 10

config={
    	"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	"theta": tune.uniform(0.01, 1.0),   # consider quniform may be better
    	"mu": tune.uniform(0.01, 1.0),
    	"step0": tune.uniform(0.01, 1.0), #[tune.uniform(0.01, 1.0) for x in range(order_range)],
        "step1": tune.uniform(0.01, 1.0),
        "step2": tune.uniform(0.01, 1.0),
        "step3": tune.uniform(0.01, 1.0),
        "step4": tune.uniform(0.01, 1.0),
        "step5": tune.uniform(0.01, 1.0),
        "step6": tune.uniform(0.01, 1.0),
        "step7": tune.uniform(0.01, 1.0),
        "step8": tune.uniform(0.01, 1.0),
        "step9": tune.uniform(0.01, 1.0),
        "sample": tune.loguniform(100,2000),
        "q": tune.randint(1,6),
    }

points_to_evaluate = [
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0,"sample":100,"q":1},
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0,"sample":500,"q":1},
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0,"sample":1000,"q":1},
    {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0,"sample":2000,"q":1},
]
evaluated_rewards = [25.48,26.85,27.48,27.52]


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
    q = config['q']
    sample_ratio = config['sample']
    cmd = f"/usr/bin/time -v numactl -i all ../LightNE -walksperedge 10000 -walklen 10 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 4096 -dim 128 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 0 -sample_ratio {sample_ratio} -upper 0 -tablesz 6679660000 -sparse_project 0 -power_iteration {q} -oversampling 50 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    macro_f1_mean = node_classification.main(['--label',label,'--embedding',pro_out,'--start-train-ratio','10','--stop-train-ratio','90','--num-train-ratio','9','--C','10','--num-split','3','--binary'])
    print("macro_f1_mean:",macro_f1_mean)
    tune.report(macro_f1_mean=macro_f1_mean)


np.random.seed(1)
analysis = tune.run(
    evaluation_function=objective,
    config = config,
    metric="macro_f1_mean",
    mode = "max",
    low_cost_partial_config={"sample":100,"q":1},
    points_to_evaluate=points_to_evaluate,
    evaluated_rewards=evaluated_rewards,
    num_samples=100, 
    #metric_constraints=[("macro_f1_mean", ">=", 27.52)],
    use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


