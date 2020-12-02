

"""
Final Model
"""
from abc import ABC
import lib
import torch as t
import numpy as np
# import matplotlib.pyplot as plt
import alexlib.toolbox as tb


#%% Load data

device = t.device(['cpu', 'cuda:0', 'cuda:1'][2])
batch_size = 32
my_writer = lib.SummaryWriter(comment='CDAN')

x_source = np.load('./data/source/processed/S_sourceAll.npy').astype(np.float32)
y_source = np.load('./data/source/processed/SourceLabels.npy').astype(np.float32)
x_target = np.load('./data/target/processed/S_targetAll.npy').astype(np.float32)
y_target = np.load('./data/target/processed/TargetLabels.npy').astype(np.float32)
D = t.tensor(np.load('./data/source/processed/distance.npy').astype(np.float32)).to(device)

print('x_source = ', x_source.shape, '  x_target = ', x_target.shape)
print('y_source = ', y_source.shape, '  y_target = ', y_target.shape)

x_source_train, x_source_test, y_source_train, y_source_test = lib.train_test_split(x_source, y_source, test_size=0.1)
x_target_train, x_target_test, y_target_train, y_target_test = lib.train_test_split(x_target, y_target, test_size=0.2)

# Convert everything to torch tensor, and move it to device.
data = 'x_source_train, x_source_test, y_source_train, y_source_test, x_target_train, x_target_test,' \
       ' y_target_train, y_target_test'.replace(' ', '').split(',')
for adata in data:
    exec(f'{adata} = t.tensor({adata}).to(device)')

dataset_source = lib.TensorDataset(x_source_train, y_source_train)
dataset_target = lib.TensorDataset(x_target_train, y_target_train)
dataloader_source = lib.DataLoader(dataset_source, batch_size=batch_size, num_workers=0)
dataloader_target = lib.DataLoader(dataset_target, batch_size=batch_size, num_workers=0)


#%% Estiamte the distance between the two distributions.

x_source = t.tensor(x_source)
y_source = t.tensor(y_source)
lib.MMD.gaussian_kernel(x_source, y_source)
loss = lib.MMD()
loss(x_source, y_source)


#%% Defining a NN


class FE(t.nn.Module, ABC):
    def __init__(self):
        super(FE, self).__init__()
        self.FEconv1 = lib.Cconv1d(in_channels=30, out_channels=60, kernel_size=2, stride=1, padding=2)
        self.FEconv2 = lib.Cconv1d(in_channels=60, out_channels=120, kernel_size=3, stride=2, padding=0)
        self.FElin1 = lib.Clinear(6360, 500)

    def forward(self, x):
        re, im = x[Ellipsis, 0], x[Ellipsis, 1]
        op0 = lib.crelu(*self.FEconv1(re, im))
        op1re, op1im = lib.crelu(*self.FEconv2(*op0))
        op2 = op1re.view(x.shape[0], -1), op1im.view(x.shape[0], -1)
        code = lib.crelu(*self.FElin1(*op2))
        return code


class LP(t.nn.Module, ABC):
    def __init__(self):
        super(LP, self).__init__()
        self.LPlin1 = lib.Clinear(500, 200)
        self.LPlin2 = lib.Clinear(200, 31)
        self.LPsm = t.nn.Softmax(dim=1)

    def forward(self, code):
        op4 = lib.crelu(*self.LPlin1(*code))
        op5 = self.LPlin2(*op4)
        label = self.LPsm(lib.zlogit(*op5))
        return label


class DC(t.nn.Module, ABC):
    def __init__(self):
        super(DC, self).__init__()
        self.DClin1 = lib.Clinear(500, 200)
        self.DClin2 = lib.Clinear(200, 2)
        self.DCsm = t.nn.LogSoftmax(dim=1)
        # self.RL1 = ReverseLayer.apply
        # self.RL2 = ReverseLayer.apply

    def forward(self, code):
        # new_code = self.RL1(code[0], alpha), self.RL2(code[1], alpha)
        op7 = lib.crelu(*self.DClin1(*code))
        op8 = lib.crelu(*self.DClin2(*op7))
        domain = self.DCsm(lib.zlogit(*op8))
        return domain


class CDan(t.nn.Module, ABC):
    def __init__(self):
        super(CDan, self).__init__()
        self.FE = FE()
        self.DC = DC()
        self.LP = LP()

    def forward(self, x):
        code = self.FE(x)
        domain = self.DC(code)
        label = self.LP(code)
        return label, domain


a = CDan()

print(a(t.randn(20, 30, 105, 2))[0].shape, a(t.randn(20, 30, 105, 2))[1].shape)


#%% Pretraining

my_net = CDan().to(device)
print(f'Number of parameters {sum(p.numel() for p in my_net.parameters())}')
# For the purpose of pretraining, we will pass parameters of the LP and FE to the optimizer, but not DC.
adam = lib.optim.Adam(list(my_net.FE.parameters()) + list(my_net.LP.parameters()), lr=0.001)
sgd = lib.optim.SGD(list(my_net.FE.parameters()) + list(my_net.LP.parameters()), lr=0.001)


def pretrain(model=my_net, opt=adam, epochs=5):
    for an_epoch in range(epochs):
        total_loss = t.tensor(0.0)
        for index, (x, y) in enumerate(dataloader_source):
            predicted_label, predicted_domain = model(x)
            loss = lib.my_kl(predicted_label, y)  # + 0.1 * cluster_dist(predicted, D)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach()
            if index % 50 == 0:
                print('Loss = ', loss.item(), end='\r')
            # writer.add_scalar('Loss', loss, global_step=next(batch_counter))

        print(f'Epoch = {an_epoch}, loss = {total_loss / len(dataloader_source):1.3f},', end=' ')
        model.eval()
        with t.no_grad():
            predicted_label, predicted_domain = model(x_source_test)
            loss = lib.my_kl(predicted_label, y_source_test)
        model.train()
        print(f'Loss on test set = {loss.item():1.3f}')
        # my_writer.add_scalar('test_loss', loss, global_step=next(epoch_counter))
    # Finally, after pretraining, let's see how good the LP on Target. And how good is the DC.
    with t.no_grad():
        predicted_label, predicted_domain = model(x_target_test)
        loss = lib.my_kl(predicted_label, y_target_test)
    print(f'Loss on Target = {loss.item():1.3f}')
    print(f'DC accuracy on Target = {lib.Accuracy()(predicted_domain, t.ones(len(x_target_test), dtype=t.long).to(device))}')
    t.save(model.state_dict(), './checkpoints/pretrained')
    print('Model was saved @ ./checkpoints/pretrained')


#%% DA trainnig

def load_pretrained():
    my_net = CDan().to(device)
    tmp = t.load('./checkpoints/pretrained')
    my_net.load_state_dict(tmp)
    return my_net


my_net = load_pretrained()
dc_metric = lib.Accuracy()
sgd = lib.optim.SGD(list(my_net.FE.parameters()) + list(my_net.LP.parameters()), lr=0.001)  # pass all parameters
adam = lib.optim.Adam(list(my_net.FE.parameters()) + list(my_net.LP.parameters()), lr=0.001)  # pass all parameters
dc_opt = lib.optim.Adam(my_net.DC.parameters(), lr=0.001)  # pass all parameters
loss_domain = t.nn.NLLLoss().to(device)
num_batches = len(dataloader_source)
epoch_counter, batch_counter = lib.count(1, 1), lib.count(1, 1)  # start counting from 1, and increment by 1


def train(model=my_net, opt1=adam, opt2=dc_opt, epochs=5, annealing=True, alpha=1.0, writer=my_writer):
    alpha = t.tensor(alpha, dtype=t.float32).to(device)
    best_perf = 1.5
    for an_epoch in range(epochs):
        dc_metric.reset_states()  # At the begining of each epoch, reset the states.
        total_loss = t.tensor(0.0)  # We use this as a metric for LP performance.
        epoch_index = next(epoch_counter)
        for index, ((x_s, y_s), (x_t, y_t)) in enumerate(zip(dataloader_source, lib.cycle(dataloader_target))):
            batch_index = next(batch_counter)
            if annealing:
                p = float(index + an_epoch * num_batches) / epochs / num_batches
                alpha = t.tensor(2. / (1. + np.exp(-10 * p)) - 1, dtype=t.float32).to(device)

            # This method to generate domain labels is dynamic and is immune to variable batch size in both datasets:
            source_domain = t.zeros(len(y_s), dtype=t.long).to(device)
            target_domain = t.ones(len(y_t), dtype=t.long).to(device)

            predicted_label_s, predicted_domain_s = model(x_s)
            predicted_label_t, predicted_domain_t = model(x_t)
            loss_lp_s = lib.my_kl(predicted_label_s, y_s)  # + 0.1 * cluster_dist(predicted, D)
            loss_lp_t = lib.my_kl(predicted_label_t, y_t)  # + 0.1 * cluster_dist(predicted, D)
            loss_dc = loss_domain(predicted_domain_s, source_domain) + loss_domain(predicted_domain_t, target_domain)
            loss_adv = loss_domain(predicted_domain_s, 1-source_domain) +\
                       loss_domain(predicted_domain_t, 1-target_domain)
            loss = loss_lp_t + loss_lp_s * 0.5 + loss_adv * alpha
            model.zero_grad()
            loss.backward(retain_graph=True)
            opt1.step()
            loss_dc.backward(retain_graph=False)
            opt2.step()

            total_loss += loss.detach()
            dc_metric(t.cat([predicted_domain_s, predicted_domain_t], dim=0),
                      t.cat([source_domain, target_domain], dim=0))

            if index % 10 == 0:  # every other while print something to show the progress
                print(f'Batch {index}/{num_batches}, LP Loss={loss_lp_s.item():1.3f}, DC Loss={loss_dc.item():1.3f}, '
                      f'DC acc={dc_metric.result():1.1f}, alpha={alpha.item():1.3f}', end='\r')
                writer.add_scalar('Inst Loss', loss, global_step=batch_index)
                writer.add_scalar('Inst LP loss S', loss_lp_s, global_step=batch_index)
                writer.add_scalar('Inst LP loss T', loss_lp_t, global_step=batch_index)
                writer.add_scalar('Inst confusion loss', loss_adv, global_step=batch_index)
                writer.add_scalar('Inst DC Loss', loss_dc, global_step=batch_index)
                writer.add_scalar('Alpha', alpha, global_step=batch_index)
                # writer.add_histogram('DC zlogits', t.cat([predicted_domain_s, predicted_domain_t]), global_step=batch_index)

        # After each epoch, print the average total loss
        print(f'Epoch {an_epoch}/{epochs}, Avg LP Loss={total_loss / num_batches:1.3f},', end=' ')
        writer.add_scalar('Total loss', total_loss / num_batches, global_step=epoch_index)

        # And check the performance on x_source_test:
        model.eval()  # Go to evaluation mode
        with t.no_grad():  # don't waste time creating graph
            predicted_label, predicted_domain = model(x_source_test)
            loss = lib.my_kl(predicted_label, y_source_test)
        print(f'LP@unseen S={loss.item():1.3f},', end=' ')
        writer.add_scalar('LP unseen S', loss, global_step=epoch_index)

        # And also check the performance on target test set.
        with t.no_grad():
            predicted_label, predicted_domain = model(x_target_test)
            loss = lib.my_kl(predicted_label, y_target_test)

        model.train()  # Back to train mode.
        if loss.item() < best_perf:  # Out top goal is to perform best on loss from target domain test set.
            best_perf = loss.item()
            t.save(model.state_dict(), f'./checkpoints/best')

        print(f'LP@unseen T={loss.item():1.3f}, DC acc={dc_metric.result():1.0f}, alpha={alpha.item():1.3f}, '
              f'best={best_perf}')
        writer.add_scalar('LP @ unseen T', loss, global_step=epoch_index)


#%% results for paper: load a saved model.

results_path = tb.temp("manuscript_figures")


def load_danned():
    model = CDan().to(device)
    tmp = t.load('./ResultModels/danned')
    model.load_state_dict(tmp)
    x_target_test_ = t.load(f'./ResultModels/da_target_x')
    y_target_test_ = t.load(f'./ResultModels/da_target_y')
    return model, x_target_test_, y_target_test_


my_net, x_target_test, y_target_test = load_danned()
with t.no_grad():
    predicted_labels, predicted_domain = my_net(x_target_test)
kld_loss = lib.my_kl(predicted_labels, y_target_test)
cxe_loss = lib.cxe(predicted_labels, y_target_test)
print(kld_loss, cxe_loss)  # this should give the performance declared in the paper. 0.0695


#%% Load saved predictions:

predicted_labels, y_target_test = np.load('./ResultModels/da_target_predict_np.npy'),\
                                  np.load('./ResultModels/da_target_y_np.npy')
predicted_labels, y_target_test = t.tensor(predicted_labels), t.tensor(y_target_test)


#%% Estimation of discrepancy between source domain and target domain.

my_net, _, _ = load_danned()

x_source = t.tensor(x_source)
x_target = t.tensor(x_target)
loss = lib.MMD(bw=1)  # Maximum Mean Discrepency loss

# discrepency before feature extractor
loss_before = loss(x_source[:1000], x_target)  # should give 0.6328

# discrepency after feature extractor
code_source_real, code_source_imag = my_net.FE(x_source[:1000].to(device))
code_source = t.stack([code_source_real, code_source_imag], dim=2)
code_target_real, code_target_imag = my_net.FE(x_target.to(device))
code_target = t.stack([code_target_real, code_target_imag], dim=2)
loss_after = loss(code_source, code_target)  # should give 0.0259

# Domain classifier quality after training. Pass 100 codes of source and target.
metric = lib.Accuracy()
res_t = my_net.DC((code_target_real[:100], code_target_imag[:100]))
res_s = my_net.DC((code_source_real[:100], code_source_imag[:100]))
source_domain = t.zeros(100, dtype=t.long).to(device)
target_domain = t.ones(100, dtype=t.long).to(device)
metric(t.cat([res_s, res_t], dim=0), t.cat([source_domain, target_domain], dim=0))  # should give 59%
