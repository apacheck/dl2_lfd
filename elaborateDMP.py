#%%
import torch
from torch import nn, optim, autograd
from ltl_diff import constraints, oracle
import os
from os.path import join
from nns.dmp_nn import DMPNN
from dmps.dmp import load_dmp_demos, DMP
from helper_funcs.conversions import np_to_pgpu
from helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from loadAndChartDMP import chart_multidim_DMP
import time

def rollout_error(output_roll, target_roll):
    return torch.norm(output_roll - target_roll, dim=2).mean()





#%%
def training_loop(train_set, val_set, constraint, enforce_constraint, adversarial, results_folder, main_loss_weight=1.0,
                  constraint_loss_weight=5.0, basis_fs=30, dt=0.01, n_epochs=200, output_dimension=6, previous_model_path=None, output_model_path=None):
    in_dim = train_set[0][0].numel()
    model = DMPNN(in_dim, 1024, output_dimension, basis_fs)
    if previous_model_path:
        model.load_state_dict(torch.load(previous_model_path))
    optimizer = optim.Adam(model.parameters())
    loss_fn = rollout_error
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    if val_set is not None:
        val_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    else:
        val_loader = None

    train_losses, val_losses = [], []

    def batch_learn(data_loader, enf_c, adv, optimize=False):
        """ Performs the learning for the weights of the dmp

        Takes in the constraints and data and performs learning. Accesses the opt and model loaded in the outside
        function. Returns the loss, learned rollouts, and a mask of the rollouts that satisfy the constraint

        Args:
            data_loader:
            enf_c:
            adv:
            optimize:

        Return:
            Losses are:
                [0] main_loss = how close it is to the old trajectories
                [1] constraint_loss = how it satisfies the ltl constraint
                [2] full_loss = combination of main_loss and constraint_loss
                [3] What percent of trajectories satisfy the ltl constraint
            learned_rollouts
            constraint_satisfaction
        """
        losses = []
        learned_rollouts_out = None
        c_sat_out = None
        for batch_idx, (starts, rollouts) in enumerate(data_loader):
            #print("Model Params {} \n".format(list(model.parameters())))
            batch_size, T, dims = rollouts.shape
            # print("B: {}".format(batch_idx))

            learned_weights = model(starts)
            dmp = DMP(basis_fs, dt, dims)
            learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]

            main_loss = loss_fn(learned_rollouts, rollouts)

            if constraint is None:
                c_loss, c_sat = torch.tensor([0]), torch.ones([starts.shape[0]])
            else:
                c_loss, c_sat = oracle.evaluate_constraint(
                    starts, rollouts, constraint, model, dmp.rollout_torch, adv)

            if enf_c:
                full_loss = main_loss_weight * main_loss + constraint_loss_weight * c_loss
            else:
                full_loss = main_loss

            losses.append([main_loss.item(), c_loss.item(), full_loss.item(), np.mean(c_sat.cpu().detach().numpy())])

            if learned_rollouts_out is None:
                learned_rollouts_out = learned_rollouts.cpu().detach().numpy()
                c_sat_out = c_sat.cpu().detach().numpy().astype(bool)
            else:
                learned_rollouts_out = np.stack([learned_rollouts_out, learned_rollouts.cpu().detach().numpy()])
                c_sat_out = np.stack([c_sat_out, c_sat.cpu().detach().numpy().astype(bool)])

            if optimize:
                optimizer.zero_grad()
                full_loss.backward()
                optimizer.step()

        return np.mean(losses, 0, keepdims=True), learned_rollouts_out, c_sat_out

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train loop
        model.train()
        avg_train_loss, learned_rollouts_epoch, c_sat_epoch = batch_learn(train_loader, enforce_constraint, adversarial, True)
        train_losses.append(avg_train_loss[0])

        # Validation Loop
        if val_loader is not None:
            model.eval()
            avg_val_loss, _, _ = batch_learn(val_loader, True, False, False)
            val_losses.append(avg_val_loss[0])

            print("e{}\t t: {}\t v: {}\t time: {}".format(epoch, avg_train_loss[0, :], avg_val_loss[0, :], time.time() - epoch_start))
        else:
            print("e{}\t t: {} time: {}".format(epoch, avg_train_loss[0, :2]), time.time() - epoch_start)

        np.save(join(results_folder, "learned_rollouts_epoch" + str(epoch) + ".txt"), learned_rollouts_epoch)
        np.save(join(results_folder, "c_sat_epoch" + str(epoch) + ".txt"), c_sat_epoch)

        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
            
    torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))
    if output_model_path is not None:
        torch.save(model.state_dict(), output_model_path)
    np.savetxt(join(results_folder, "train_losses.txt"), train_losses)
    np.savetxt(join(results_folder, "val_losses.txt"), val_losses)
    return model


def evaluate_constraint(val_set, constraint, model_path, basis_fs=30, dt=0.01, n_epochs=200, output_dimension=6):
    """ Evaluates a given constraint for a set of inputs given a model

    Args:
        val_set:
        constraint:
        model_path:
        basis_fs:
        dt:
        n_epochs:
    Return:
        losses (constraint loss, percent satisfying constraint)
        learned rollouts
        constraint satisfaction for rollout
    """
    in_dim = val_set[0][0].numel()
    model = DMPNN(in_dim, 1024, output_dimension, basis_fs)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    learned_rollouts_out = None
    c_sat_out = None
    losses = []
    for batch_idx, (starts, rollouts) in enumerate(val_loader):
        batch_size, T, dims = rollouts.shape

        learned_weights = model(starts)
        dmp = DMP(basis_fs, dt, dims)
        learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]

        if constraint is None:
            c_loss, c_sat = torch.tensor([0]), torch.ones([starts.shape[0]])
        else:
            c_loss, c_sat = oracle.evaluate_constraint(
                starts, rollouts, constraint, model, dmp.rollout_torch, False)
        losses.append([c_loss.item(), np.mean(c_sat.cpu().detach().numpy())])

        if learned_rollouts_out is None:
            learned_rollouts_out = learned_rollouts.cpu().detach().numpy()
            c_sat_out = c_sat.cpu().detach().numpy().astype(bool)
        else:
            learned_rollouts_out = np.stack([learned_rollouts_out, learned_rollouts.cpu().detach().numpy()])
            c_sat_out = np.stack([c_sat_out, c_sat.cpu().detach().numpy().astype(bool)])

    return np.mean(losses, 0, keepdims=True), learned_rollouts_out, c_sat_out


#%%
# def chartDMP():
    # # Visualization stuff (can probably be in separate file / functions...)
    # %matplotlib auto
    # # plt.style.use('seaborn')
    # model.eval()

    # train_starts, train_rollout = train_set[0]
    # val_y0 = train_starts[0].unsqueeze(0)
    # val_goal = train_starts[-1].unsqueeze(0)

    # learned_weights = model(train_starts.unsqueeze(0))

    # dmp = DMP(basis_fs, dt, t_pose_hists.shape[2])
    # learned_dmp_rollout = dmp.rollout_torch(val_y0, val_goal, learned_weights, 1.0)[0][0].detach().cpu()

    # learned_displacements =  learned_dmp_rollout[1:, :] - learned_dmp_rollout[:-1, :]
    # learned_velocities = torch.norm(learned_displacements, dim=1)
    # print(learned_velocities)
    # # displacements = torch.zeros_like(rollout)
    # # displacements[:, 1:, :] = rollout[:, 1:, :] - rollout[:, :-1, :] # i.e., v_t = x_{t + 1} - x_t
    # # velocities = ltd.TermDynamic(torch.norm(displacements, dim=2, keepdim=True))

    # # dmp_timescale = np.linspace(0, 1, learned_dmp_rollout.shape[0])

    # fig, ax = plt.subplots()
    # ax.scatter(train_rollout.detach().cpu()[:, 0], train_rollout.detach().cpu()[:, 1], label="Demo", c='orange', alpha=0.5)
    # ax.scatter(learned_dmp_rollout[:, 0], learned_dmp_rollout[:, -1], label="DMP + LTL", alpha=0.5)

    # # ax.scatter(train_starts[1:3, 0].detach().cpu(),
    # #             train_starts[1:3, 1].detach().cpu(),
    # #             c='r', marker='x', s=10**2)

    # # ax.plot([0.0, 1.0], [0.75, 0.75], c='r')
    # # ax.plot([0.0, 1.0], [0.25, 0.25], c='r')
    # # ax.scatter(train_starts[-1, 0].detach().cpu(),
    # #             train_starts[-1, 1].detach().cpu(),
    # #             c='r', marker='x')
    # # ax.scatter(train_starts[3, 0].detach().cpu(),
    # #             train_starts[3, 1].detach().cpu(),
    # #             c='black', marker='x')
    # # ax.add_patch(plt.Circle(train_starts[1], radius=0.1, color="red", alpha=0.1))

    # # plt.xlabel("X")
    # plt.xlim(0.0, 1.0)
    # # plt.ylabel("Y")
    # plt.ylim(0.0, 1.0)
    # plt.legend(prop={"size": 14})
    # plt.tight_layout()
    # plt.show()
    # Load the start states and pose hists



# For each demo, for each instance, for each variation of trained with / without constraint....
# demo_constraints = {
#     "avoid": constraints.AvoidPoint(1, 0.1, 1E-2),
#     "patrol": constraints.EventuallyReach([1, 2], 1E-2),
#     "stable": constraints.StayInZone(
#                 torch.tensor([0.0, 0.25], device=torch.device("cuda")),
#                 torch.tensor([1.0, 0.75], device=torch.device("cuda")),
#                 1E-2),
#     "slow": constraints.MoveSlowly(0.01, 1E-2),
#     "redCubeReach-flat": constraints.AvoidAndPatrolConstant(
#         torch.tensor([0.15, 0.5]).cuda(),
#         torch.tensor([0.5, 0.5]).cuda(),
#         0.1,
#         1E-2),
#     "wobblyPour-flat" : constraints.DontTipEarly(
#         torch.tensor([0.0, -np.pi, -np.pi]).cuda(),
#         torch.tensor([0.2, np.pi, np.pi]).cuda(),
#         2, 0.00,  1e-2)
#     # "wobblyPour-flat": constraints.StayInZone(
#     #     torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.6]).cuda(),
#     #     torch.tensor([1.0, 1.0, 1.0, 0.75, 0.05, -0.5]).cuda(),
#     #     1e-2
#     # )
# }

"""
# Robot Single Demo Loop:
demo_types = ["wobblyPour-flat"]
demo_types = ["redCubeReach-flat"]
demo_types = ["wobblyPour-flat", "redCubeReach-flat"]
results_root = "logs/robot-exps-{}".format(t_stamp())
enforcement_types = ["unconstrained", "train"]
# enforcement_types = ["train"]
for demo_type in demo_types:
    for enforce_type in enforcement_types:
        if enforce_type == "unconstrained":
            enforce = False
        else:
            enforce = True

        print("{}, {}".format(demo_type, enforce_type))
        demo_folder = "demos/{}".format(demo_type)

        t_start_states, t_pose_hists = load_dmp_demos(demo_folder + "/train")
        t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
        train_set = TensorDataset(t_start_states[0:1], t_pose_hists[0:1])

        # chart_multidim_DMP(demo_folder + "/train", "")

        results_folder = join(results_root, "{}-{}".format(demo_type, enforce_type))
        os.makedirs(results_folder, exist_ok=True)

        constraint = demo_constraints[demo_type]

        learned_model = training_loop(train_set, None, constraint, enforce, False, results_folder)
"""


"""
# Generalized Demonstration Loop:
demo_types = ["avoid", "patrol", "slow", "stable"]
enforcement_types = ["unconstrained", "train", "adversarial"]
results_root = "logs/generalized-exps-{}".format(t_stamp())
for demo_type in demo_types:
    for enforce_type in enforcement_types:
        if enforce_type == "unconstrained":
            enforce, adversarial = False, False
        if enforce_type == "train":
            enforce, adversarial = True, False
        if enforce_type == "adversarial":
            enforce, adversarial = True, True
        print("{}, {}".format(demo_type, enforce_type))
        demo_folder = "demos/{}".format(demo_type)

        t_num = 100
        t_start_states, t_pose_hists = load_dmp_demos(demo_folder + "/train")
        t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
        train_set = TensorDataset(t_start_states[0:t_num], t_pose_hists[0:t_num])

        v_num = 20
        v_start_states, v_pose_hists = load_dmp_demos(demo_folder + "/val")
        v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
        val_set = TensorDataset(v_start_states[0:v_num], v_pose_hists[0:v_num])

        results_folder = join(results_root, "{}-{}".format(demo_type, enforce_type))
        os.makedirs(results_folder, exist_ok=True)

        constraint = demo_constraints[demo_type]

        learned_model = training_loop(train_set, val_set, constraint, enforce, adversarial, results_folder)
"""




# # Single Demonstration Loop
# for demo_type in ["avoid", "patrol", "slow", "stable"]:
#     for i in range(5):
#         for enforce_constraint in [True, False]:
#             print("{}, {}, {}".format(demo_type, i, enforce_constraint))
#             demo_folder = "demos/{}".format(demo_type)
#
#             t_start_states, t_pose_hists = load_dmp_demos(demo_folder + "/train")
#             t_start_states = np_to_pgpu(t_start_states)
#             t_pose_hists = np_to_pgpu(t_pose_hists)
#             train_set = TensorDataset(t_start_states[i:i+1], t_pose_hists[i:i+1])
#
#             # Create time-stamped results folder TODO: (possibly move this out to a util func?)
#             results_folder_variable = "logs/single-shot-experiments/{}-{}-enforce{}".format(demo_type, i, enforce_constraint)
#             os.makedirs(results_folder_variable, exist_ok=True)
#
#             constraint = demo_constraints[demo_type]
#
#             learned_model = training_loop(train_set, None, constraint, enforce_constraint, False, results_folder_variable)
