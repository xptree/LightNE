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

logging.basicConfig(filename="./tune_mag.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "/home/jiezhong/dev/graph_embedding/data_bin"
ne_out = "mag.ne"
pro_out = "mag.pro"
label = f"{data_path}/mag.label.npz"
input_path = f"{data_path}/mag.adj"
order_range = 10
#'coeff0': 1.4848048177980857, 'coeff1': 24.232865772919947, 'coeff2': 89.84839953900072, 'coeff3': 39.06190944850238,'coeff4': 16.813765523550977, 'coeff5': 79.70530983657791, 'coeff6': 42.834033922771376, 'coeff7': 26.87509553313273, 'coeff8': 99.45812727875744, 'coeff9': 14.927623734817985, 'order': 11.0, 'theta': 0.779052477013066, 'mu': 0.11351752202853674
#35.96 (0.00)
#18.73 (0.00)

config={
    	"order": tune.randint(5, 16),   # half open randint like np.random.randint, right side is open
    	"theta": tune.uniform(0.1, 1.0),   # consider quniform may be better
    	"mu": tune.uniform(0.1, 1.0),
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
        "q": tune.randint(1,6),
    }
points_to_evaluate = [
        {"order": 10, "theta": 0.5, "mu":0.2, "step0":1.0, "step1":1.0,"step2":1.0,"step3":1.0,"step4":1.0,"step5":1.0,"step6":1.0,"step7":1.0,"step8":1.0,"step9":1.0,"q":5},
]
evaluated_rewards = [24.77]

#search_alg = flaml.BlendSearch(space = config,
#        metric = "macro_f1_mean",
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
    order = config['order']
    theta = config['theta']
    mu = config['mu']
    q = config['q']
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 1 -walklen 10 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 256 -dim 128 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 1 -sample_ratio 0.077 -mem_ratio 0.5 -upper 0 -tablesz 34179869184 -sparse_project 0 -power_iteration {q} -oversampling 0 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)
    macro_f1_mean = node_classification.main(['--label',label,'--embedding',pro_out,'--start-train-ratio','0.001','--stop-train-ratio','0.001','--num-train-ratio','1','--C','10','--num-split','1','--binary'])
    print(macro_f1_mean)
    tune.report(macro_f1_mean=macro_f1_mean)

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
    metric = "macro_f1_mean",
    mode="max",
    points_to_evaluate=points_to_evaluate,
    evaluated_rewards=evaluated_rewards,
    num_samples=50, use_ray=False)

print("best config and result:\n")
print(analysis.best_config)
print(analysis.best_trial.last_result)


#search_alg.save("./checkpoints/mag.tune")



