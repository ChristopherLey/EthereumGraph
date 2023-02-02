import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from FraudDetection.datareaders import get_elliptic_graph
from FraudDetection.GATv2 import GATv2Conv

elliptic_data = get_elliptic_graph(Path("/data/Datastore/Elliptic"))
print(f"{elliptic_data.train_idx}{elliptic_data.valid_idx}{elliptic_data.test_idx}")


class TransactionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(TransactionClassifier, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=args["heads"])
        self.conv2 = GATv2Conv(
            args["heads"] * hidden_dim, hidden_dim, heads=args["heads"]
        )

        self.post_mp = nn.Sequential(
            nn.Linear(args["heads"] * hidden_dim, hidden_dim),
            nn.Dropout(args["dropout"]),
            nn.Linear(hidden_dim, output_dim),
        )

        self.dropout = nn.Dropout(args["dropout"])

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.dropout(F.relu(x))
        # MLP output
        x = self.post_mp(x)
        return torch.sigmoid(x)


class MetricManager(object):
    def __init__(self, modes=["train", "val"]):

        self.output = {}

        for mode in modes:
            self.output[mode] = {}
            self.output[mode]["accuracy"] = []
            self.output[mode]["f1micro"] = []
            self.output[mode]["f1macro"] = []
            self.output[mode]["aucroc"] = []
            # new
            self.output[mode]["precision"] = []
            self.output[mode]["recall"] = []
            self.output[mode]["cm"] = []

    def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):

        # calculate metrics
        pred_labels = pred_scores > threshold
        accuracy = accuracy_score(target_labels, pred_labels)
        f1micro = f1_score(target_labels, pred_labels, average="micro")
        f1macro = f1_score(target_labels, pred_labels, average="macro")
        aucroc = roc_auc_score(target_labels, pred_scores)
        # new
        recall = recall_score(target_labels, pred_labels)
        precision = precision_score(target_labels, pred_labels)
        cm = confusion_matrix(target_labels, pred_labels)

        # Collect results
        self.output[mode]["accuracy"].append(accuracy)
        self.output[mode]["f1micro"].append(f1micro)
        self.output[mode]["f1macro"].append(f1macro)
        self.output[mode]["aucroc"].append(aucroc)
        # new
        self.output[mode]["recall"].append(recall)
        self.output[mode]["precision"].append(precision)
        self.output[mode]["cm"].append(cm)

        return accuracy, f1micro, f1macro, aucroc, recall, precision, cm

        # Get best results

    def get_best(self, metric, mode="val"):

        # Get best results index
        best_results = {}
        i = np.array(self.output[mode][metric]).argmax()

        # Output
        for m in self.output[mode].keys():
            best_results[m] = self.output[mode][m][i]

        return best_results


class GnnTrainer(object):
    def __init__(self, model):
        self.model = model
        self.metric_manager = MetricManager(modes=["train", "val"])
        self.data_train = None

    def train(self, data_train, optimizer, criterion, scheduler, args):

        self.data_train = data_train
        for epoch in range(args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data_train)

            out = out.reshape((data_train.x.shape[0]))
            loss = criterion(
                out[data_train.train_idx], data_train.y[data_train.train_idx]
            )
            # Metric calculations
            # train data
            target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
            (
                train_acc,
                train_f1,
                train_f1macro,
                train_aucroc,
                train_recall,
                train_precision,
                train_cm,
            ) = self.metric_manager.store_metrics("train", pred_scores, target_labels)

            # Training Step
            loss.backward()
            optimizer.step()

            # validation data
            self.model.eval()
            target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
            (
                val_acc,
                val_f1,
                val_f1macro,
                val_aucroc,
                val_recall,
                val_precision,
                val_cm,
            ) = self.metric_manager.store_metrics("val", pred_scores, target_labels)

            if epoch % 5 == 0:
                print(
                    "epoch: {} - loss: {:.4f} - accuracy train: {:.4f} - "
                    "accuracy valid: {:.4f}  - val roc: {:.4f}  - val f1micro: {:.4f}".format(
                        epoch, loss.item(), train_acc, val_acc, val_aucroc, val_f1
                    )
                )

    # To predict labels
    def predict(self, data=None, unclassified_only=True, threshold=0.5):
        # evaluate model:
        self.model.eval()
        if data is not None:
            self.data_train = data

        out = self.model(self.data_train)
        out = out.reshape((self.data_train.x.shape[0]))

        if unclassified_only:
            pred_scores = out.detach().cpu().numpy()[self.data_train.test_idx]
        else:
            pred_scores = out.detach().cpu().numpy()

        pred_labels = pred_scores > threshold

        return {"pred_scores": pred_scores, "pred_labels": pred_labels}

    # To save metrics
    def save_metrics(self, save_name, path="./save/"):
        file_to_store = open(path + save_name, "wb")
        pickle.dump(self.metric_manager, file_to_store)
        file_to_store.close()

    # To save model
    def save_model(self, save_name, path="./save/"):
        torch.save(self.model.state_dict(), path + save_name)


# Set training arguments, set prebuild=True to use builtin PyG models otherwise False
args = {
    "epochs": 300,
    "lr": 0.01,
    "weight_decay": 1e-5,
    "prebuild": True,
    "heads": 2,
    "hidden_dim": 128,
    "dropout": 0.5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransactionClassifier(
    elliptic_data.num_node_features, args["hidden_dim"], 1, args
)
model.double().to(device)
data_train = elliptic_data.to(device)

# Setup training settings
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
criterion = torch.nn.BCELoss()

# Train
gnn_trainer_gat = GnnTrainer(model)
gnn_trainer_gat.train(data_train, optimizer, criterion, scheduler, args)

gnn_trainer_gat.save_metrics("GATv2.results", path="./save_results/")
gnn_trainer_gat.save_model("GATv2.pth", path="./save_results/")


# model.load_state_dict(torch.load("./save_results/" + "GATv2prebuilt.pth"))
# gnn_t2 = GnnTrainer(model)
# output = gnn_t2.predict(data=data_train, unclassified_only=False)
#
# time_period = 28
# sub_node_list = df_merge.index[df_merge.loc[:, 1] == time_period].tolist()
#
# # Fetch list of edges for that time period
# edge_tuples = []
# for row in data_train.edge_index.view(-1, 2).to("cpu").numpy():
#   if (row[0] in sub_node_list) | (row[1] in sub_node_list):
#     edge_tuples.append(tuple(row))
# len(edge_tuples)
#
# # Fetch predicted results for that time period
# node_color = []
# for node_id in sub_node_list:
#   if node_id in classified_illicit_idx: #
#      label = "red" # fraud
#   elif node_id in classified_licit_idx:
#      label = "green" # not fraud
#   else:
#     if output['pred_labels'][node_id]:
#       label = "orange" # Predicted fraud
#     else:
#       label = "blue" # Not fraud predicted
#
#   node_color.append(label)
#
# # Setup networkx graph
# G = nx.Graph()
# G.add_edges_from(edge_tuples)
#
# # Plot the graph
# plt.figure(3,figsize=(16,16))
# plt.title("Time period:"+str(time_period))
# nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)
