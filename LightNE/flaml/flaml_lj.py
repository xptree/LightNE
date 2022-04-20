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

logging.basicConfig(filename="./tune_lj.log", filemode="a", level=logging.INFO,format='%(asctime)s %(message)s')

data_path = "../../datasets"
ne_out = "lj.ne_v2"
pro_out = "lj.prone_v2"
input_path = f"{data_path}/lj.adj"
order_range = 5
#'coeff0': 40.075846267350876, 'coeff1': 92.11202925693883, 'coeff2': 77.86030297853927, 'coeff3': 12.70243815507488, 'coeff4': 0.04332500267602768, 'order': 10.0, 'theta': 0.9788035917096728, 'mu': 0.36935297093793507

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
evaluated_rewards = [26.85]

search_alg = flaml.BlendSearch(space = config,
        metric = "score",
        mode = "max",
        #points_to_evaluate=points_to_evaluate,
        #evaluated_rewards=evaluated_rewards
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
    cmd = f"/usr/bin/time -v numactl -i all ./LightNE -walksperedge 1 -walklen 5 -step_coeff {step_coeff} -rounds 1 -s -m -ne_out {ne_out} -pro_out {pro_out} -ne_method netsmf -rank 1024 -dim 1024 -order {order} -theta {theta} -mu {mu} -analyze 1 -sample 1 -sample_ratio 13.0 -mem_ratio 1 -upper 0 -tablesz 15494752578 -sparse_project 0 -power_iteration {q} -oversampling 10 {input_path}"
    logger.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,stderr = p.communicate()
    logger.info(out)
    logger.info(stderr)

    x = np.fromfile("/home/xieyy/Graph_Embedding/LightNE/LightNE/lj.prone_v2", dtype=np.float32).reshape(-1, 1024)
    hf = h5py.File("/home/xieyy/Graph_Embedding/LightNE/LightNE/model/livejournal/embeddings_user_id_0.v30.h5", mode='a')
    hf["embeddings"][:, :] = x
    hf.close()

    subprocess.call('./eval.sh')
    float_array = np.load('/home/xieyy/Graph_Embedding/LightNE/LightNE/metrics.npy')
    logger.info(float_array)
    score=((float_array[1]+float_array[2]+float_array[3]+float_array[4]+float_array[5])/5)
    #print(average_score)
    tune.report(score=score)

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


search_alg.save("./checkpoints/lj.tune")



