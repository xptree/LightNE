#!/usr/bin/env python
# encoding: utf-8

import scipy.io
import math
import numpy as np
import pathlib
import scipy.sparse as sp
import argparse
import logging
import pandas as pd

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s') # include timestamp
logger = logging.getLogger(__name__)

def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    return data[variable_name]

def load_edgelist(files, relabel=False, make_sym=False, idmap="idmap.npy", pbg_idmap=""):
    logger.info("loading edgelist")
    data_list = [pd.read_csv(file, sep='\t', engine='c', comment='#', header=None).to_numpy()
            for file in files]
    data = data_list[0]
    logger.info("data.shape=(%d, %d)", data.shape[0], data.shape[1])
    logger.info("load edgelist done, %d edges", data.shape[0])
    diag = data[:, 0] == data[:, 1]
    data = data[~diag]
    m = data.shape[0]
    logger.info("after removing self-loops, %d edges left", m)

    if relabel:
        if len(pbg_idmap):
            import rapidjson
            with open(pbg_idmap, "r") as f:
                to_idx, from_idx = zip(*enumerate(map(int, rapidjson.load(f))))
            print(to_idx[:10])
            print(from_idx[:10])
        else:
            to_idx, from_idx = zip(*enumerate(np.unique(np.concatenate(data_list))))
        logger.info("create from_idx and to_idx, %d unique vertex idx", len(to_idx))
        to_idx, from_idx = np.array(to_idx), np.array(from_idx)
        logger.info("to numpy")
        mapping = -np.ones(np.max(from_idx) + 1, dtype=int)
        mapping[from_idx] = to_idx
        logger.info("create mapping")
        data = mapping[data]
        logger.info("map done")
        np.save(idmap, mapping)

    n = np.max(data) + 1
    logger.info("%d nodes, %d edges" % (n, m))

    #  label = None
    #  if label_file:
    #      #  label = np.loadtxt(label_file, dtype=int, max_rows=None)
    #      label = pd.read_csv(label_file, sep='\t', engine='c', comment='#', header=None).to_numpy()
    #      logger.info("label.shape=(%s)", ", ".join(map(str, label.shape)))
    #      if relabel:
    #          for i in range(label.ndim-1):
    #              label[:, i] = mapping[label[:, i]]
    #              map_check_res = np.all(label[:, i] >= 0)
    #          assert map_check_res, 'idx in label file not appear in edgelist file!!!'
    #      to_label, from_label = zip(*enumerate(np.unique(label[:, 1])))
    #      logger.info("create from_label and to_label, %d unique labels", len(to_label))
    #      to_label, from_label = np.array(to_label), np.array(from_label)
    #      logger.info("to numpy")
    #      label_mapping = -np.ones(np.max(from_label) + 1, dtype=int)
    #      label_mapping[from_label] = to_label
    #      logger.info("create label mapping")
    #      label[:, -1] = label_mapping[label[:, -1]]
    #      logger.info("label map done")
    val = [1.] * (2*m if make_sym else m)
    i = np.concatenate((data[:, 0], data[:, 1]), axis=None) if make_sym else data[:, 0]
    j = np.concatenate((data[:, 1], data[:, 0]), axis=None) if make_sym else data[:, 1]

    return sp.coo_matrix((val, (i, j)), shape=(n, n))

def x2adjcencygraph(files, output, idmap, relabel=False, make_sym=False, pbg_idmap=""):
    file = files[0]
    suffix = pathlib.Path(file).suffix
    if suffix == '.mat':
        logger.info("mat2adj from %s to %s" % (file, output))
        A = load_adjacency_matrix(file)
        assert A.shape[0] == A.shape[1], "should be square matrix"
        A = A.todok()
        for i in range(A.shape[0]):
            A[i, i] = 0
        A = A.tocoo()
        A.eliminate_zeros()
        min_v, max_v = min(A.data) , max(A.data)
        logger.info("minimum non-zero value=%.2f maximum non-zero value=%.2f" \
                % (min_v, max_v))
        unweighted = math.isclose(min_v, 1.0) and math.isclose(max_v, 1.0)
    elif suffix == '.edge' or suffix == '.txt' or suffix == '.csv':
        logger.info("edgelist2adj from %s to %s" % (file, output))
        A = load_edgelist(files, relabel=relabel, make_sym=make_sym, idmap=idmap, pbg_idmap=pbg_idmap)
        #  if label is not None:
        #      np.save(label_output, label)
        #      logger.info("dumped label file to %s", label_output)
        unweighted = True
    else:
        raise NotImplementedError

    A = A.tocsr()
    sym_err = A - A.T
    sym_check_res = np.all(np.abs(sym_err.data) < 1e-10)  # tune this value
    assert sym_check_res, 'input matrix is not symmetric!!!'
    logger.info("dumping AdjacencyGraph to %s", output)
    with open(output, "w") as f:
        print("AdjacencyGraph" if unweighted else "WeightedAdjacencyGraph", end="\n", file=f)
        print(A.shape[0], end='\n', file=f)
        print(A.nnz, end='\n', file=f)
        print("\n".join(map(str, A.indptr.tolist()[:-1])), end="\n", file=f)
        print("\n".join(map(str, A.indices.tolist())), end="" if unweighted else "\n", file=f)
        if not unweighted:
            print("\n".join(map(lambda x: str(int(x)), A.data.tolist())), end="", file=f)
    logger.info("dump done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("to AdjacencyGraph")
    parser.add_argument("--file", type=str, required=True, action='append',
            help="input file, only the first file will be converted to AdjacencyGraph, others are used for building idmap")
    parser.add_argument("--output", type=str, default="graph.adj", help="output adj graph file")
    parser.add_argument("--idmap", type=str, default="idmap.npy", help="idmap")
    parser.add_argument("--relabel", action="store_true", help="relabel input edgelist")
    parser.add_argument("--pbg-idmap", type=str, default="", help="idmap from pbg (.json)")
    parser.add_argument("--make-sym", action="store_true", help="make input graph symmetric")
    args = parser.parse_args()
    print(args)
    x2adjcencygraph(args.file, args.output, args.idmap, relabel=args.relabel, make_sym=args.make_sym, pbg_idmap=args.pbg_idmap)
