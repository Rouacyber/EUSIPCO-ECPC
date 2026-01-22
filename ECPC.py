import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from Functions import train_and_evaluate
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.optim as optim
import numpy as np
import torch.optim as optim
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()

def relu_evidence(logits):
    return F.relu(logits)

def exp_evidence(logits): 
    return torch.exp(torch.clamp(logits, -10, 10))

def softplus_evidence(logits):
    return F.softplus(logits)

# Functions for KL divergence and evidence
def KL(alpha, K):
    beta = torch.ones((1, K), dtype=torch.float32)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

# MSE loss for EDL
def mse_loss(target, alpha, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    A = torch.sum((target - p) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1.0, global_step / annealing_step)
    return A + annealing_coef * B

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Should be a tensor of class weights if provided
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def focal_edl_mse_loss(one_hot_target, alpha, gamma=2.0):
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    mse = torch.sum((one_hot_target - probs) ** 2, dim=1)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)
    base_loss = mse + var  # shape: [batch]

    # Focal weighting: modulate based on confidence in true class
    pt = torch.sum(probs * one_hot_target, dim=1)  # model's confidence in true class
    focal_weight = (1.0 - pt) ** gamma

    return focal_weight * base_loss  # still shape: [batch]

def train_epoch(model, train_loader, optimizer, global_step, K, annealing_step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch.float32), target.to(torch.int64)
        optimizer.zero_grad()
        logits, evidence = model(data)
        alpha = evidence + 1
        u = K / torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        one_hot_target = F.one_hot(target, num_classes=K).to(torch.float32)
        loss = torch.mean(mse_loss(one_hot_target, alpha, global_step, annealing_step))
        l2_loss =  (torch.norm(model.fc1.weight) + torch.norm(model.fc2.weight)) * 0.005
        total_loss = loss + l2_loss
        total_loss.backward()
        optimizer.step()
        global_step += 1  # Increment global step
    return global_step

#for imbalanced datasets
def train_epoch_imb(model, train_loader, optimizer, global_step, K, annealing_step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch.float32), target.to(torch.int64)
        optimizer.zero_grad()

        logits, evidence = model(data)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        one_hot_target = F.one_hot(target, num_classes=K).to(torch.float32)

        # Replace base EDL loss with focal-modulated version
        loss = torch.mean(focal_edl_mse_loss(one_hot_target, alpha, gamma=2.0))

        # Keep L2 regularization exactly the same
        l2_loss = (torch.norm(model.fc1.weight) + torch.norm(model.fc2.weight)) * 0.005
        total_loss = loss + l2_loss

        total_loss.backward()
        optimizer.step()
        global_step += 1

    return global_step

def evaluate(model, loader, K):
    model.eval()
    correct = 0
    total_samples = 0
    all_predictions = []
    all_probs = []
    uncertainties = []
    #targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(torch.float32), target.to(torch.int64)
            logits, evidence = model(data)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
            u = K / alpha_sum
            pred = logits.argmax(dim=1)
            #print(target, pred)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
            #targets.append(target.cpu())
            all_predictions.append(pred.cpu())
            all_probs.append(prob.cpu())
            uncertainties.append(u)
    #print(correct)
    #print(total_samples)
    #stop
    acc = correct / total_samples
    all_predictions = torch.cat(all_predictions).numpy()
    #targets = torch.cat(targets).numpy()
    #print(accuracy_score(targets ,all_predictions))
    all_probs = torch.cat(all_probs)
    uncertainties = torch.cat(uncertainties)
    return acc, all_predictions, all_probs, uncertainties

def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs, K, annealing_step):
    global_step = 0
    train_acc_list = []
    test_acc_list = []
    train_predictions_list = []
    test_predictions_list = []
    #train_probs_list = []
    #test_probs_list = []
    train_labels_list = []  # To store the labels for the training set
    test_labels_list = []  # To store the labels for the testing set

    for epoch in range(num_epochs):
        global_step = train_epoch(model, train_loader, optimizer, global_step, K, annealing_step)
    
        # Evaluate on training data
        train_acc, train_predictions, train_probs, train_uncerts = evaluate(model, train_loader, K)
        train_labels = torch.cat([target for _, target in train_loader]).cpu().numpy()
    
        # Evaluate on testing data
        test_acc, test_predictions, test_probs, test_uncerts = evaluate(model, test_loader, K)
        test_labels = torch.cat([target for _, target in test_loader]).cpu().numpy()

        # Store results
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_predictions_list.append(train_predictions)
        test_predictions_list.append(test_predictions)

        # Compute precision, recall, F1 scores
        train_precision = precision_score(train_labels, train_predictions, zero_division=0)
        test_precision = precision_score(test_labels, test_predictions, zero_division=0)
        train_recall = recall_score(train_labels, train_predictions)
        test_recall = recall_score(test_labels, test_predictions)
        train_f1 = f1_score(train_labels, train_predictions)
        test_f1 = f1_score(test_labels, test_predictions)

    return train_acc, test_acc, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, train_predictions, test_predictions, train_probs, test_probs

def train_and_evaluate_mult(model, train_loader, test_loader, optimizer, num_epochs, K, annealing_step):
    global_step = 0
    train_acc_list = []
    test_acc_list = []
    train_predictions_list = []
    test_predictions_list = []
    train_probs_list = []
    test_probs_list = []
    for epoch in range(num_epochs):
        global_step = train_epoch(model, train_loader, optimizer, global_step, K, annealing_step)  # Train model for one epoch

        # Evaluate on training data
        train_acc, train_predictions, train_probs, train_uncerts = evaluate(model, train_loader, K)
        train_labels = torch.cat([target for _, target in train_loader]).cpu().numpy()

        # Evaluate on testing data
        test_acc, test_predictions, test_probs, test_uncerts = evaluate(model, test_loader, K)
        test_labels = torch.cat([target for _, target in test_loader]).cpu().numpy()

        # Store results
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_predictions_list.append(train_predictions)
        test_predictions_list.append(test_predictions)
        #train_probs_list.append(train_probs)
        #test_probs_list.append(test_probs)
        """print(train_labels)
        print(train_predictions)
        stop"""
        # Print progress
        #print(f"Epoch {epoch+1}:Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
        train_precision_per_class = precision_score(train_labels, train_predictions, average=None, zero_division=0)
        test_precision_per_class = precision_score(test_labels, test_predictions, average=None, zero_division=0)
        #print("Train Precision per Class:", train_precision_per_class)

        #train_acc = accuracy_score(train_labels, train_predictions)
        #test_acc = accuracy_score(test_labels, test_predictions)
        
        train_recall = recall_score(train_labels, train_predictions, average=None, zero_division=0)
        test_recall = recall_score(test_labels, test_predictions, average=None, zero_division=0)

        train_f1 = f1_score(train_labels, train_predictions, average=None, zero_division=0)
        test_f1 = f1_score(test_labels, test_predictions, average=None, zero_division=0)

    return train_acc, test_acc, train_precision_per_class, test_precision_per_class, train_recall, test_recall, train_f1, test_f1, train_predictions, test_predictions, train_probs, test_probs#, train_uncerts, test_uncerts

def train_epoch_test(model, train_loader, optimizer, global_step, K, annealing_step):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch.float32), target.to(torch.int64)
        optimizer.zero_grad()
        logits, evidence = model(data)
        alpha = evidence + 1
        one_hot_target = F.one_hot(target, num_classes=K).to(torch.float32)
        loss = torch.mean(mse_loss(one_hot_target, alpha, global_step, annealing_step))
        l2_loss = (torch.norm(model.fc1.weight) + torch.norm(model.fc2.weight)) * 0.005
        total_batch_loss = loss + l2_loss
        total_batch_loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += total_batch_loss.item()

    average_loss = total_loss / len(train_loader)
    return global_step, average_loss

def evaluate_test(model, loader, K, global_step, annealing_step):
    model.eval()
    correct = 0
    total_samples = 0
    all_predictions = []
    all_probs = []
    uncertainties = []
    total_loss = 0.0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(torch.float32), target.to(torch.int64)
            logits, evidence = model(data)
            alpha = evidence + 1
            one_hot_target = F.one_hot(target, num_classes=K).to(torch.float32)

            # MSE loss
            loss = torch.mean(mse_loss(one_hot_target, alpha, global_step, annealing_step))
            total_loss += loss.item()

            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / alpha_sum
            u = K / alpha_sum
            pred = logits.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
            all_predictions.append(pred.cpu())
            all_probs.append(prob.cpu())
            uncertainties.append(u)

    acc = correct / total_samples
    all_predictions = torch.cat(all_predictions).numpy()
    all_probs = torch.cat(all_probs)
    uncertainties = torch.cat(uncertainties)

    avg_loss = total_loss / len(loader)
    return acc, all_predictions, all_probs, uncertainties, avg_loss

def train_and_evaluate_to_test_epochs(model, train_loader, test_loader, optimizer, num_epochs, K, annealing_step):
    global_step = 0
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
        global_step, train_loss = train_epoch_test(model, train_loader, optimizer, global_step, K, annealing_step)
        train_acc, train_predictions, train_probs, train_uncerts, train_eval_loss = evaluate_test(model, train_loader, K, global_step, annealing_step)
        test_acc, test_predictions, test_probs, test_uncerts, test_eval_loss = evaluate_test(model, test_loader, K, global_step, annealing_step)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_eval_loss)
        test_loss_list.append(test_eval_loss)

    return {
        "train_acc_list": train_acc_list,
        "test_acc_list": test_acc_list,
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list
    }

class TabularLeNet(nn.Module):
    def __init__(self, num_features, num_classes, evidence_func=exp_evidence):
        super(TabularLeNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = (num_features - 6) // 4  
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.evidence_func = evidence_func

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))  
        x = F.max_pool1d(x, kernel_size=2)  
        x = F.relu(self.conv2(x)) 
        x = F.max_pool1d(x, kernel_size=2)  
        x = x.flatten(1)  
        x = F.relu(self.fc1(x))  
        logits = self.fc2(x)  
        evidence = self.evidence_func(logits)
        return logits, evidence

class ConvNet(nn.Module):
    def __init__(self, batch_size, input_channels, signal_len, p, evidence_func=exp_evidence):
        super(ConvNet, self).__init__()
        self.drop = nn.Dropout(p)
        self.conv1 = nn.Conv1d(input_channels,64, 1)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.conv3 = nn.Conv1d(32, 16, 1)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, signal_len)
        self.evidence_func = evidence_func

    def forward(self, x):

      x = x.unsqueeze(2)
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      #x = self.drop(x)
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.drop(x)
      x = self.fc2(x)
      evidence = self.evidence_func(x)
      return x, evidence
        
def get_calibration_probabilities(model, calibration_loader):
    model.eval()
    all_calibration_probs = []

    with torch.no_grad():
        for data, _ in calibration_loader:
            data = data.to(device)
            logits, evidence = model(data)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            all_calibration_probs.append(prob.cpu())
    
    all_calibration_probs = torch.cat(all_calibration_probs)
    return all_calibration_probs

def compute_prediction_sets(probs, qhat):

    return probs >= (1 - qhat)

def compute_true_labels_sets(prediction_sets, labels):

    true_labels_sets = []
    
    for row in prediction_sets:
        true_indices = np.where(row)[0]  # Find indices where class is predicted
        true_labels_sets.append(true_indices)
    
    return true_labels_sets

def compute_coverage(true_labels_sets, true_labels):

    assert len(true_labels_sets) == len(true_labels), "Length mismatch between prediction sets and true labels"
    correct = 0
    true_labels_list = true_labels.tolist()
    
    for i in range(len(true_labels)):
        prediction = true_labels_sets[i]
        if true_labels_list[i] in prediction:
            correct += 1   
    percentage_correct = (correct / len(true_labels)) * 100

    return percentage_correct #(correct / len(true_labels)) * 100

def compute_average_length(prediction_sets):

    return np.mean([len(p) for p in prediction_sets])

def compute_empty_sets_percentage(prediction_sets):
    count = 0
    for p in prediction_sets:
        if isinstance(p, np.ndarray) and p.size == 0:
            count += 1
    return count #(count / len(prediction_sets)) * 100

def compute_p_values(scores, cal_scores):
    
    scores = np.array(scores)
    counts = np.zeros_like(scores)
    p_values = np.zeros_like(scores, dtype=float)

    for i, score in enumerate(scores):
        count_greater = np.sum(cal_scores >= score)
        p_values[i] = (count_greater + 1) / (len(cal_scores) + 1)

    return p_values#, np.array(p_values)

def compute_mean_p_value(scores, cal_scores):
    
    p_values = compute_p_values(scores, cal_scores)
    return np.mean(p_values)

def compute_entropy(probs):

    #return -torch.sum(probs * torch.log2(probs + 1e-12), dim=1).mean().item()
    return -torch.sum(probs * torch.log2(probs + 1e-12), dim=1)
    
def compute_entropy_mean(probs):

    return -torch.sum(probs * torch.log2(probs + 1e-12), dim=1).mean().item()

def compute_brier_score(probs, targets):

    return [(probs[i][1] - targets[i]) ** 2 for i in range(len(probs))]

def BS_multiclass(probs: torch.Tensor, labels: np.ndarray) -> list:
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long, device=probs.device)

    # Convert labels to one-hot encoding
    num_classes = probs.shape[1]
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    # Compute Brier Score for each sample
    brier_scores = torch.sum((probs - labels_one_hot) ** 2, dim=1)
    return brier_scores.cpu().numpy()

def compute_brier_score_mean(probs, targets):

    return np.mean([(probs[i][1] - targets[i]) ** 2 for i in range(len(probs))])

def compute_brier_score_mult_mean(probs, targets):

    return np.mean(BS_multiclass(probs, targets))

def evaluate_prediction_sets(probs, labels, cal_scores, qhat):

    # Compute prediction sets
    prediction_sets = compute_prediction_sets(probs, qhat)
    true_labels_sets = compute_true_labels_sets(prediction_sets, labels)
    # Compute metrics
    coverage = compute_coverage(true_labels_sets, labels)
    avg_length = compute_average_length(true_labels_sets)

    scores = []
    for i, image in enumerate(labels):
        smx = probs[i]
        #smx = probs.numpy()
        score = 1 - smx[labels[i]]  # Extract the score for the corresponding label
        #scores = 1 - smx[np.arange(len(labels)), labels]
        scores.append(score)
    mean_p_value = compute_mean_p_value(scores, cal_scores)
    
    oneC = np.mean([len(p) == 1 for p in true_labels_sets]) * 100

    return {
        "coverage": coverage,
        "avg_length": avg_length,
        "mean_p_value": mean_p_value,
        "oneC": oneC
    }
    
def evaluate_prediction_sets_ood(probs, labels, cal_scores, qhat):

    prediction_sets = compute_prediction_sets(probs, qhat)
    true_labels_sets = compute_true_labels_sets(prediction_sets, labels)

    avg_length = compute_average_length(true_labels_sets)

    scores = []
    for i, image in enumerate(labels):
        smx = probs[i]
        #smx = probs.numpy()
        score = 1 - smx[labels[i]]  # Extract the score for the corresponding label
        #scores = 1 - smx[np.arange(len(labels)), labels]
        scores.append(score)
    mean_p_value = compute_mean_p_value(scores, cal_scores)

    oneC = np.mean([len(p) == 1 for p in true_labels_sets]) * 100

    return {
        "avg_length": avg_length,
        "mean_p_value": mean_p_value,
        "oneC": oneC
    }

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def average_metric_dicts(dicts):
    # Get keys from first dict
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts]) for k in keys}

def multiple_runs_with_noisy_ood(
    model, train_loader, test_loader, calibration_loader, 
    noisy_loader, ood_loader,
    optimizer_class, num_epochs, K, num_runs
):
    annealing_step = 10 * len(test_loader)
    all_run_results = []

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")

        # Reinitialize model and optimizer for each run
        model.apply(reset_weights) 
        model = model.to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.001)
        #train_accuracy = []
        #test_accuracy = []
        # Train over epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            tr_acc, test_acc, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, \
            train_predictions, test_predictions, train_probs, test_probs = train_and_evaluate(
                model, train_loader, test_loader, optimizer, 1, K, annealing_step
            )
        #test_acc = np.mean(test_accuracy)
        # Calibration
        y_train = train_loader.dataset.tensors[1].numpy()
        y_test = test_loader.dataset.tensors[1].numpy()
        y_cal = calibration_loader.dataset.tensors[1].numpy()

        cal_probs = get_calibration_probabilities(model, calibration_loader)
        cal_smx = cal_probs.numpy()
        n = len(cal_smx)
        cal_scores = 1 - cal_smx[np.arange(n), y_cal]

        alpha = 0.1
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, method='higher')

        # Train/Test Metrics
        test_metrics = evaluate_prediction_sets(test_probs, y_test, cal_scores, qhat)
        train_metrics = evaluate_prediction_sets(train_probs, y_train, cal_scores, qhat)

        sensitivity, specificity = compute_sensitivity_specificity(y_test, test_predictions)
        entropy_test = compute_entropy_mean(test_probs)
        entropy_train = compute_entropy_mean(train_probs)
        brier_test = compute_brier_score_mean(test_probs, y_test)
        brier_train = compute_brier_score_mean(train_probs, y_train)

        # ========== Noisy Evaluation ==========
        y_noisy = noisy_loader.dataset.tensors[1].numpy()
        noisy_acc, noisy_predictions, noisy_probs, noisy_uncerts = evaluate(model, noisy_loader, K)
        noisy_metrics = evaluate_prediction_sets(noisy_probs, y_test, cal_scores, qhat)
        entropy_noisy = compute_entropy_mean(noisy_probs)
        brier_noisy = compute_brier_score_mean(noisy_probs, y_test)

        # ========== OOD Evaluation ==========
        y_ood = ood_loader.dataset.tensors[1].numpy()
        ood_acc, ood_predictions, ood_probs, ood_uncerts = evaluate(model, ood_loader, K)
        ood_metrics = evaluate_prediction_sets_ood(ood_probs, y_test, cal_scores, qhat)
        entropy_ood = compute_entropy_mean(ood_probs)
        # Optional: Uncomment if Brier score is valid for OOD
        # brier_ood = compute_brier_score_mean(ood_probs, y_ood)
                
        
        ece_test = compute_ece(test_probs, y_test)
        ece_train = compute_ece(train_probs, y_train)
        ece_noisy = compute_ece(noisy_probs, y_test)  # or y_noisy if matched correctly


        # Store results for this run
        run_result = {
            "train_accuracy": tr_acc,
            "test_accuracy": test_acc,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "entropy_test": entropy_test,
            "entropy_train": entropy_train,
            "brier_test": brier_test,
            "brier_train": brier_train,
            "noisy_accuracy": noisy_acc,
            "noisy_metrics": noisy_metrics,
            "entropy_noisy": entropy_noisy,
            "brier_noisy": brier_noisy,
            "ood_accuracy": ood_acc,
            "ood_metrics": ood_metrics,
            "entropy_ood": entropy_ood,
            "ece_test": ece_test,
            "ece_train": ece_train,
            "ece_noisy": ece_noisy,

            # "brier_ood": brier_ood  # Uncomment if needed
        }

        all_run_results.append(run_result)
        print(f"Run {run + 1} completed.\n")
        plot_calibration_curve(test_probs, y_test, title=f"Test Calibration Curve (Run {run+1})")
        plot_calibration_curve(train_probs, y_train, title=f"Train Calibration Curve (Run {run+1})")
        plot_calibration_curve(noisy_probs, y_test, title=f"Noisy Calibration Curve (Run {run+1})")  # or y_noisy

    # Return full results for later averaging/analysis
    return all_run_results

def multiple_runs_with_noisy_mult(
    model, train_loader, test_loader, calibration_loader, 
    noisy_loader,
    optimizer_class, num_epochs, K, num_runs
):
    annealing_step = 10 * len(test_loader)
    all_run_results = []

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")

        # Reinitialize model and optimizer for each run
        model.apply(reset_weights)  # Helper function (below)
        model = model.to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.001)
        #train_accuracy = []
        #test_accuracy = []
        # Train over epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            tr_acc, test_acc, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, \
            train_predictions, test_predictions, train_probs, test_probs = train_and_evaluate_mult(
                model, train_loader, test_loader, optimizer, num_epochs, K, annealing_step
            )
        #test_acc = np.mean(test_accuracy)
        y_train = train_loader.dataset.tensors[1].numpy()
        y_test = test_loader.dataset.tensors[1].numpy()
        y_cal = calibration_loader.dataset.tensors[1].numpy()

        cal_probs = get_calibration_probabilities(model, calibration_loader)
        cal_smx = cal_probs.numpy()
        n = y_cal.shape[0]
        #n = len(cal_smx)
        cal_scores = 1 - cal_smx[np.arange(n), y_cal]

        alpha = 0.1
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, method='higher')

        # Train/Test Metrics
        test_metrics = evaluate_prediction_sets(test_probs, y_test, cal_scores, qhat)
        train_metrics = evaluate_prediction_sets(train_probs, y_train, cal_scores, qhat)

        #sensitivity, specificity = compute_sensitivity_specificity(y_test, test_predictions)
        entropy_test = compute_entropy_mean(test_probs)
        entropy_train = compute_entropy_mean(train_probs)
        brier_test = compute_brier_score_mean(test_probs, y_test)
        brier_train = compute_brier_score_mean(train_probs, y_train)

        # ========== Noisy Evaluation ==========
        y_noisy = noisy_loader.dataset.tensors[1].numpy()
        noisy_acc, noisy_predictions, noisy_probs, noisy_uncerts = evaluate(model, noisy_loader, K)
        noisy_metrics = evaluate_prediction_sets(noisy_probs, y_test, cal_scores, qhat)
        entropy_noisy = compute_entropy_mean(noisy_probs)
        brier_noisy = compute_brier_score_mean(noisy_probs, y_test)

        # Store results for this run
        run_result = {
            "train_accuracy": tr_acc,
            "test_accuracy": test_acc,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            #"sensitivity": sensitivity,
            #"specificity": specificity,
            "entropy_test": entropy_test,
            "entropy_train": entropy_train,
            "brier_test": brier_test,
            "brier_train": brier_train,
            "noisy_accuracy": noisy_acc,
            "noisy_metrics": noisy_metrics,
            "entropy_noisy": entropy_noisy,
            "brier_noisy": brier_noisy,
        }

        all_run_results.append(run_result)
        print(f"Run {run + 1} completed.\n")

    # Return full results for later averaging/analysis
    return all_run_results