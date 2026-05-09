from implements.wl_aksvd import WL_AKSVD
from implements.wl_bayesian import WL_BAYESIAN
from utils.graph_data import GraphDataLoader
from sklearn.model_selection import train_test_split
data_loader = GraphDataLoader()

def main():
    print("Running all implementations...")

    graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

    G_train, G_test, y_train, y_test = train_test_split(
        graphs, y,
        test_size=0.2,
        random_state=42
    )

    G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
        G_train, y_train,
        test_size=0.75,
        random_state=42
    )

    wl_aksvd = WL_AKSVD(data_loader)
    print(f"Running {wl_aksvd.implementation}...")
    wl_aksvd.run(G_vocab_train, G_ML_train, G_test, y_ML_train, y_test)

    wl_bayesian = WL_BAYESIAN(data_loader)
    print(f"Running {wl_bayesian.implementation}...")
    wl_bayesian.run(G_vocab_train, G_ML_train, G_test, y_ML_train, y_test)

if __name__ == "__main__":
    main()