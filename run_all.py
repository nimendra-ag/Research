from implements.wl_aksvd import WL_AKSVD
from implements.wl_bayesian import WL_BAYESIAN
from implements.wl_csfddl import WL_CSFDDL
from implements.wl_fddl import WL_FDDL
from utils.evaluator import Evaluator
from utils.graph_data import GraphDataLoader

data_loader = GraphDataLoader()

def main():
    # print("Running all implementations...")

    # wl_aksvd = WL_AKSVD(data_loader)
    # print(f"Running {wl_aksvd.implementation}...")
    # wl_aksvd.run()

    # wl_bayesian = WL_BAYESIAN(data_loader)
    # print(f"Running {wl_bayesian.implementation}...")
    # wl_bayesian.run()
    
    wl_fddl = WL_FDDL(data_loader)
    print(f"Running {wl_fddl.implementation}...")
    wl_fddl.run()
    
    wl_csfddl = WL_CSFDDL(data_loader)
    print(f"Running {wl_csfddl.implementation}...")
    wl_csfddl.run()

if __name__ == "__main__":
    main()